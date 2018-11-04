import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import random
import time

from dataset import SOS_token, EOS_token, Seq2SeqDataset
from dataset import tensors_from_pair, prepare_data
from utils import time_since, show_plot
from model import AttnDecoderRNN, EncoderRNN
from evaluate import evaluate_randomly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 256
teacher_forcing_ratio = 0.5


def train_iteration(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_func):
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    encoder_hidden = encoder.init_hidden().to(device)
    encoder_outputs = torch.zeros(decoder.max_length, encoder.hidden_size, device=device)
    
    # Zero the model gradients.
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    
    # Encoder.
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Decoder.
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

            loss += loss_func(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

            _, top_index = decoder_output.topk(1)
            decoder_input = top_index.squeeze().detach()  # detach from history as input

            loss += loss_func(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train(encoder, decoder, n_iters, dataloader, input_lang, output_lang, print_every=1000, plot_every=1000, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    loss_func = nn.NLLLoss()
    iter_ = 1
    for input_tensor, target_tensor in enumerate(dataloader):

        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        loss = train_iteration(input_tensor, target_tensor, encoder,
                               decoder, encoder_optimizer, decoder_optimizer, loss_func)
        print_loss_total += loss
        plot_loss_total += loss

        if iter_ % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter_ / n_iters),
                                         iter_, iter_ / n_iters * 100, print_loss_avg))

        if iter_ % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    return plot_losses


def main():
    input_lang, output_lang, pairs = prepare_data('eng', 'fra', reverse=False)

    seqdataset = Seq2SeqDataset(pairs, input_lang, output_lang)
    seqdataloader = DataLoader(seqdataset, batch_size=2)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, max_length=10).to(device)

    loss_history = []
    for i in range(5):
        losses = train(encoder, attn_decoder, len(pairs), seqdataloader, input_lang=input_lang, output_lang=output_lang, print_every=1000)

        loss_history.extend(losses)
        evaluate_randomly(encoder, attn_decoder, pairs, max_length=10, input_lang=input_lang, output_lang=output_lang)

    show_plot(loss_history)
    print('done training')


if __name__ == "__main__":
    main()

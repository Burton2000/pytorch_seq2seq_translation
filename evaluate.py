import torch
import random
from dataset import SOS_token, EOS_token
from dataset import tensor_from_sentence
from utils import show_attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(encoder, decoder, sentence, max_length, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden().to(device)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            _, top_index = decoder_output.data.topk(1)
            if top_index.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[top_index.item()])

            decoder_input = top_index.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluate_randomly(encoder, decoder, pairs, max_length, input_lang, output_lang, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], max_length, input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def evaluate_and_show_attention(input_sentence, encoder, attn_decoder, max_length, input_lang, output_lang):
    output_words, attentions = evaluate(
        encoder, attn_decoder, input_sentence, max_length, input_lang, output_lang)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)

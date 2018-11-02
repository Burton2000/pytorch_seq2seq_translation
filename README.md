## Sentence translation with seq2seq models

In this repo I am trying out sequence to sequence modelling using encoder/decoder rnns with attention in PyTorch. The task is sentence translation between 2 languages (currently English and French).
  
I have adapated this from the tutorial provided on the PyTorch website (check refernce)

### Example outputs
This is some outputs from the trained model. For these following examples the first line is the English input sentence, the second is the French ground truth and the third line is the output from the trained model.  

**Input:** you re a jolly good feller .  
**Label:** vous etes un joyeux drille .  
**Model output:** t es un joyeux drille .  

**Input:** you are beautiful .  
**Label:** vous etes beaux .  
**Model output:** vous etes beau .  

**Input:** he is a dreamer .  
**Label:** il est reveur .  
**Model output:** il est reveur .  


#### Reference 
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-decoder

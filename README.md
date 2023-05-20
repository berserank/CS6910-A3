# CS6910-A3
Using recurrent neural networks to build a transliteration system.
The goal of this assignment is threefold: 
(i) learn how to model sequence-to-sequence learning problems using Recurrent Neural Networks 
(ii) compare different cells such as vanilla RNN, LSTM and GRU 
(iii) understand how attention networks overcome the limitations of vanilla seq2seq models

Entire code can be found in the following Kaggle Notebook : https://www.kaggle.com/code/adityanandakishore/cs6910-a3-ipynb

Wandb Report link : https://api.wandb.ai/links/berserank/sqi3bf5s

## Dataset
I have bulit the model using Aksharantar's Eng-Tamil Dataset. Data has been Pre-processed to account for variable length and unknwon characters in test data set. I have impleneted padding to account for the former and Loss was calculated over padding too. This won't affect the results as later seen in attention map. 

## Question 2- Buliding and Training Seq2Seq Model

I have implemented the entire code in a Kaggle Notebook.Here is the link for the same

https://www.kaggle.com/code/adityanandakishore/cs6910-a3-ipynb

For evaluation purposes, I have made a model.py file, that can help with the same. model.py contains the following functions. Please note that I have implemented wandb only in Kaggle Notebook but not in model.py, you will be able to see the results, but won't be able to log it.

- trainIters : This function will train the model based on the given hyper-parameters. Setting Attention = False will not be necessary as it is the default value. This function gives out val accuracy as the output after training the entire model
- infer : This function infers the built model over test data and stores the predictions in the given file(given log = True)

To deal with different number of layers in encoder and decoder, I have initialised all the hidden states of decoder with last layer of the encoder only. 


Hyper parameter Sweep was implemented in the kaggle notebook over the following hyper-parameters

- Optimiser: Nadam, Adam
- Teacher forcing ratio: 0.3, 0.5, 0.7
- Encoder Embedding: 128, 256
- Decoder Embedding: 128, 256
- Epochs: 5, 8
- Hidden Size: 128, 256, 512
- Encoder Layers: 2, 3
- Decoder Layers: 1, 2, 3
- Dropout: 0.25, 0.4
- Cell Type: GRU, RNN, LSTM
- Bidirectional: True, False (Please note that "null" indicates "Bidirectional = True" too as it is the default value)

Results and comments are attached in the wandb report : https://api.wandb.ai/links/berserank/sqi3bf5s
## Question 4- Results on Test Data

The best model reported in the above sweep had the following hyperparameters:
- Optimiser: Nadam
- Teacher forcing ratio: 0.7
- Encoder Embedding: 128
- Decoder Embedding: 256
- Epochs: 5
- Hidden Size: 512
- Encoder Layers: 3
- Decoder Layers: 1
- Dropout: 0.4
- Cell Type: Bidirectional LSTM
- Batch Size : 32
- Val Accuracy Reported - __48.14%__

The accuracy reported on test set was __33.40%__

Results and Comments are attached in the wandb report : https://api.wandb.ai/links/berserank/sqi3bf5s

## Question 5- Buliding and Training the Seq2Seq with Attention Model

The model was later trained by adding a Bahdanau attention layer to the basic sequence-to-sequence model.
Code and Reults are available in the Kaggle notebook and Wandb Report respectively.

For the sake of evaluation, one can set the "Attention Flag = True" and train the model.py. Further code specifications are given below.

I have performed a hyper parameter search again in a lesser space this time as results were satisfactory.

## Code Specifications




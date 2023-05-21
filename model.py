import csv
import numpy as np
import random
import torch
import torch.nn as nn 
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import math
import sys, getopt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def print_keys_for_values(dictionary, values):
    found_keys = []
    for value in values:
        for key, val in dictionary.items():
            if value == val:
                found_keys.append(key)
    if found_keys:
        str = ''.join(found_keys)
        if(str[-1] == '\n'):
          str = str[:-1]
        elif(str[0] == '\t'):
          str = str[1:]
    else:
        print("No keys found for the given values.")

    return str

def char_level_accuracy(targets, outputs):
    
  with torch.no_grad():
    count = 0
    total_count = 0
    for i in range(targets.shape[0]):
        same_elements = []
        for j in range(targets.shape[1]):
            if(targets[i][j] != 67 or targets[i][j] != 66):
                same_elements.append(outputs[i][j].item() == targets[i][j].item())
        count += np.sum(same_elements)
        total_count += len(same_elements)
  return count/(total_count)



def word_level_accuracy(targets, outputs):
  outputs1 = torch.argmax(outputs, dim = 1)
  with torch.no_grad():
    count = 0
    for i in range(targets.shape[0]):
      if ((outputs1[i] == targets[i]).sum().item() == targets.shape[1]):
        count = count + 1
  return count/targets.shape[0]

def sample_equidistant_points(data, epochs):
    step = len(data) // epochs
    indices = np.arange(0, len(data), step)
    equidistant_points = [data[i] for i in indices]
    
    return equidistant_points


#Data-Preprocessing

file1 = open('tam_test.csv')
file2 = open('tam_valid.csv')
file3 = open('tam_valid.csv')


csvreader1 = csv.reader(file1)
csvreader2 = csv.reader(file2)
csvreader3 = csv.reader(file3)


header1 = []
header1 = next(csvreader1)

header2 = []
header2 = next(csvreader2)

header3 = []
header3 = next(csvreader3)

test = []
val = []
train = []
for row in csvreader1:
        test.append(row)

for row in csvreader2:
        val.append(row)

for row in csvreader3:
        train.append(row)


file1.close()
file2.close()
file3.close()

def read_data(list):
  inputs = []
  targets = []
  for pair in list:
    inputs.append(pair[0])
    targets.append(pair[1])
  return inputs,targets

train_inputs, train_targets = read_data(train)
test_inputs, test_targets = read_data(test)
val_inputs, val_targets = read_data(val)

#Additional Characters

start_char = '\t'
end_char = '\n'
blank_char = ' '
unknown_char = '\r'

#Data Processing

def language_dict(inputs,targets):
  input_dict = {}
  max_input_length = 0
  input_char = []

  target_dict = {}
  max_target_length = 0
  target_char = []
  #Encoding Inputs and updating input_dict
  for string in inputs:
    max_input_length = max(len(string), max_input_length)
    for char in string:
      if char not in input_dict:
        input_dict[char] = len(input_char)
        input_char.append(char)
  if blank_char not in input_dict:
    input_dict[blank_char] = len(input_char)
    input_char.append(blank_char)
    
  input_dict[unknown_char] = len(input_char)
  input_char.append(unknown_char)
  
  if start_char not in target_dict:
    target_dict[start_char] = len(target_char)
    target_char.append(start_char)

  for string in targets:
    max_target_length = max(len(string)+2, max_target_length)
    for char in string:
      if char not in target_dict:
        target_dict[char] = len(target_char)
        target_char.append(char)

  if end_char not in target_dict:
    target_dict[end_char] = len(target_char)
    target_char.append(end_char)

  if blank_char not in target_dict:
    target_dict[blank_char] = len(target_char)
    target_char.append(blank_char)
    
  return input_dict,max_input_length,input_char,target_dict ,max_target_length,target_char

# input_dict,max_input_length,input_char,target_dict ,max_target_length,target_char = language_dict(test_inputs,test_targets)     
input_dict,max_input_length,input_char,target_dict,max_target_length,target_char = language_dict(train_inputs+val_inputs+test_inputs,train_targets+val_targets+test_targets)     

def word_to_encoding(word_list, language_dict, max_length, language):
    encodings = []
    for word in word_list:
        encoding = []
        for char in word:
            if char in language_dict:
                encoding.append(language_dict[char])
            else:
                encoding.append(language_dict[unknown_char])
        if (language == 0):
          while len(encoding) < max_length:
              encoding.append(language_dict[blank_char])
        if (language == 1):
          encoding.insert(0,language_dict[start_char])
          while len(encoding) < max_length-1:
            encoding.append(language_dict[blank_char])
          encoding.append(language_dict[end_char])
        encodings.append(encoding)
    return encodings


def process_data(train,val,test,input_dict,target_dict,max_input_length,max_target_length):

  train_inputs, train_targets = read_data(train)
  test_inputs, test_targets = read_data(test)
  val_inputs, val_targets = read_data(val)

  encoded_train_inputs = word_to_encoding(train_inputs,input_dict,max_input_length,0)
  encoded_train_targets = word_to_encoding(train_targets,target_dict,max_target_length,1)
  encoded_val_inputs = word_to_encoding(val_inputs,input_dict,max_input_length,0)
  encoded_val_targets = word_to_encoding(val_targets,target_dict,max_target_length,1)
  encoded_test_inputs = word_to_encoding(test_inputs,input_dict,max_input_length,0)
  encoded_test_targets = word_to_encoding(test_targets,target_dict,max_target_length,1)

  return encoded_train_inputs,encoded_train_targets, encoded_val_inputs, encoded_val_targets, encoded_test_inputs, encoded_test_targets


encoded_train_inputs,encoded_train_targets,encoded_val_inputs, encoded_val_targets, encoded_test_inputs, encoded_test_targets = process_data(train,val,test,input_dict,target_dict,max_input_length,max_target_length)


def convert_to_tensor_pairs(train_inputs, train_targets):
    pairs = []
    for input_data, target_data in zip(train_inputs, train_targets):
        input_tensor = torch.tensor(input_data)
        target_tensor = torch.tensor(target_data)
        pairs.append((input_tensor,target_tensor))
    return pairs

encoded_train_pairs = convert_to_tensor_pairs(encoded_train_inputs,encoded_train_targets)

encoded_val_pairs = convert_to_tensor_pairs(encoded_val_inputs,encoded_val_targets)

encoded_test_pairs = convert_to_tensor_pairs(encoded_test_inputs,encoded_test_targets)

pairs = (encoded_train_pairs, encoded_val_pairs, encoded_test_pairs)
test_pairs = pairs[2]

class EncoderRNN(nn.Module):
    def __init__(self, device, cell_type, vocab_size, embed_dim, hidden_size, num_layers=1, bidirectional=False, dropout_p=0):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)
        if(cell_type == 'lstm'):
          self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers = self.num_layers, batch_first=True, dropout = dropout_p, bidirectional=self.bidirectional)
        elif (cell_type == 'rnn'):
          self.rnn = nn.RNN(embed_dim, hidden_size, num_layers = self.num_layers, batch_first=True,dropout = dropout_p, bidirectional=self.bidirectional)
        elif (cell_type == 'gru'):
          self.rnn = nn.GRU(embed_dim, hidden_size, num_layers = self.num_layers, batch_first=True,dropout = dropout_p, bidirectional=self.bidirectional)

    def forward(self, x, hidden, cell):
        out = self.embedding(x)#.unsqueeze(1)
        out = self.dropout(out)
        if (self.cell_type == 'lstm'):
          out, (hidden, cell) = self.rnn(out, (hidden, cell))
          return out, hidden, cell
        elif (self.cell_type == 'rnn'):
          out, hidden = self.rnn(out, hidden)
          return out, hidden
        elif (self.cell_type == 'gru'):
          out, hidden = self.rnn(out,hidden) 
          return out, hidden
    
    def init_hidden(self, batch_size):
        hidden = torch.randn((1+int(self.bidirectional))*self.num_layers, batch_size, self.hidden_size, device=device)
        cell = torch.randn((1+int(self.bidirectional))*self.num_layers, batch_size, self.hidden_size, device=device)
        return hidden, cell

class DecoderRNN(nn.Module):
    def __init__(self, device, cell_type, output_vocab, embed_size, hidden_size, max_length, dropout_p=0.1, num_layers = 1, bidirectional = False):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_vocab
        self.embed_size = embed_size
        self.dropout_p = dropout_p
        self.cell_type = cell_type 
        self.max_length = max_length
        self.device = device
        self.num_layers = num_layers
        self.embedding_decoder = nn.Embedding(self.output_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.bidirectional = bidirectional


        if (cell_type == 'lstm'):
          self.rnn = nn.LSTM( self.embed_size, hidden_size, num_layers = self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout = self.dropout_p)
        elif (cell_type == 'gru'):
          self.rnn = nn.GRU( self.embed_size, hidden_size, num_layers = self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout = self.dropout_p)
        elif (cell_type == 'rnn'):
          self.rnn = nn.RNN( self.embed_size, hidden_size, num_layers = self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout = self.dropout_p)

        self.out = nn.Linear((1+int(self.bidirectional))*self.hidden_size, self.output_size)
        self.out_activation = nn.LogSoftmax(dim=-1)
  


    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded_decoder = self.embedding_decoder(input)
        embedded_decoder = self.dropout(embedded_decoder)


        if (self.cell_type == 'lstm'):
          output, (hidden, cell) = self.rnn(embedded_decoder, (hidden, cell))
        elif (self.cell_type == 'gru'):
          output, hidden = self.rnn(embedded_decoder, hidden)
        elif (self.cell_type == 'rnn'):
          output,hidden = self.rnn(embedded_decoder, hidden)
    
        output = F.relu(self.out(output))
        output = F.log_softmax(output,dim=-1)

        return output, hidden, cell

    def init_hidden(self, encoder_hidden, encoder_cell, encoder_bidirectional):
        hidden = encoder_hidden[-(1+int(encoder_bidirectional)): ].repeat(self.num_layers,1,1)
        cell = encoder_cell[-(1+int(encoder_bidirectional)): ].repeat(self.num_layers,1,1)
        return hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_target_length = 0
        self.sos = 0
        
        
    def forward(self, source, target, teacher_forcing_ratio = 0.5):
      
        batch_size = target.shape[0]
        target_len = target.shape[1]
        self.max_target_length = target_len
        target_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)

        encoder_hidden, encoder_cell = self.encoder.init_hidden(batch_size)

        
        if (self.encoder.cell_type == 'lstm'):
          encoder_outputs, encoder_hidden, encoder_cell = self.encoder.forward(source, encoder_hidden, encoder_cell)
        if (self.encoder.cell_type == 'rnn'):
          encoder_outputs, encoder_hidden = self.encoder.forward(source, encoder_hidden, encoder_cell)
        if (self.encoder.cell_type == 'gru'):
          encoder_outputs, encoder_hidden = self.encoder.forward(source, encoder_hidden, encoder_cell)


        #first input to the decoder is the <sos> tokens
        input = target[:,0]
        self.sos = target[:,0]
        hidden,cell = self.decoder.init_hidden(encoder_hidden, encoder_cell, self.encoder.bidirectional)
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder.forward(input, hidden, cell)
            outputs[:,t] = output.squeeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(-1)
            input = target[:,t] if teacher_force else top1.squeeze(1)

        return outputs

    def inference(self, source, target):
      
        batch_size = source.shape[0]
        target_len = self.max_target_length
        target_vocab_size = self.decoder.output_size
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        encoder_hidden, encoder_cell = self.encoder.init_hidden(batch_size)

        
        if (self.encoder.cell_type == 'lstm'):
          encoder_outputs, encoder_hidden, encoder_cell = self.encoder.forward(source, encoder_hidden, encoder_cell)
        if (self.encoder.cell_type == 'rnn'):
          encoder_outputs, encoder_hidden = self.encoder.forward(source, encoder_hidden, encoder_cell)
        if (self.encoder.cell_type == 'gru'):
          encoder_outputs, encoder_hidden = self.encoder.forward(source, encoder_hidden, encoder_cell)
           
        #first input to the decoder is the <sos> tokens
        input = self.sos

        hidden,cell = self.decoder.init_hidden(encoder_hidden, encoder_cell, self.encoder.bidirectional)
      
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder.forward(input, hidden, cell)
            outputs[:,t] = output.squeeze(1)
            top1 = output.argmax(-1)
            input = top1.squeeze(1)
        
        return outputs

class AttentionDecoderRNN(nn.Module):
    def __init__(self, device, cell_type, output_vocab, embed_size, hidden_size, max_length, dropout_p=0.1, num_layers = 1, bidirectional = False):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_vocab
        self.embed_size = embed_size
        self.dropout_p = dropout_p
        self.cell_type = cell_type 
        self.max_length = max_length
        self.device = device
        self.num_layers = num_layers
        self.embedding_decoder = nn.Embedding(self.output_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.bidirectional = bidirectional


        if (cell_type == 'lstm'):
          self.rnn = nn.LSTM( self.embed_size + hidden_size*(1+int(self.bidirectional)), hidden_size, num_layers = self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout = self.dropout_p)
        elif (cell_type == 'gru'):
          self.rnn = nn.GRU(self.embed_size+ hidden_size*(1+int(self.bidirectional)), hidden_size, num_layers = self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout = self.dropout_p)
        elif (cell_type == 'rnn'):
          self.rnn = nn.RNN(self.embed_size+ hidden_size*(1+int(self.bidirectional)), hidden_size, num_layers = self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout = self.dropout_p)


        self.energy = nn.Linear(hidden_size*(2+int(self.bidirectional)),hidden_size)
        self.value = nn.Linear(hidden_size,1, bias = False)
        self.softmax = nn.Softmax(dim = 0)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.out = nn.Linear((1+int(self.bidirectional))*self.hidden_size, self.output_size)
        self.out_activation = nn.LogSoftmax(dim=-1)
  
        self.hidden_reshape_linear = nn.Linear(hidden_size*2,hidden_size)


    def forward(self, input, encoder_states, hidden, cell):
        input = input.unsqueeze(1)
        embedded_decoder = self.embedding_decoder(input)
        embedded_decoder = self.dropout(embedded_decoder)

        encoder_states = encoder_states.permute(1,0,2)
        sequence_length = encoder_states.shape[0]
        if self.bidirectional == True:
          hidden_1 = self.relu(self.hidden_reshape_linear(hidden[0:2].permute(1,0,2).reshape(hidden.shape[1],-1))).unsqueeze(0)
        else:
          hidden_1 = hidden[0]

          
        hidden_reshaped = hidden_1.repeat(sequence_length,1,1)

        energy = self.value(self.tanh(self.energy(torch.cat((hidden_reshaped,encoder_states),dim=2))))
        attention = self.softmax(energy)
        attention = attention.permute(1,2,0)
        encoder_states = encoder_states.permute(1,0,2)
        context_vector = torch.bmm(attention, encoder_states)

        rnn_input = torch.cat((context_vector,embedded_decoder), dim = 2)
      
        
        if (self.cell_type == 'lstm'):
          decoder_output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        elif (self.cell_type == 'gru'):
          decoder_output, hidden = self.rnn(rnn_input, hidden)
        elif (self.cell_type == 'rnn'):
          decoder_output,hidden = self.rnn(rnn_input, hidden)

        output = F.relu(self.out(decoder_output))
        output = F.log_softmax(output,dim=-1)

        return output, hidden, cell, attention

    def init_hidden(self, encoder_hidden, encoder_cell, encoder_bidirectional):
        hidden = encoder_hidden[-(1+int(encoder_bidirectional)): ].repeat(self.num_layers,1,1)
        cell = encoder_cell[-(1+int(encoder_bidirectional)): ].repeat(self.num_layers,1,1)
        return hidden, cell

class AttentionSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_target_length = 0
        self.sos = 0
               
    def forward(self, source, target, teacher_forcing_ratio = 0.5):
      
        batch_size = target.shape[0]
        target_len = target.shape[1]
        self.max_target_length = target_len
        target_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)

        encoder_hidden, encoder_cell = self.encoder.init_hidden(batch_size)

        
        if (self.encoder.cell_type == 'lstm'):
          encoder_outputs, encoder_hidden, encoder_cell = self.encoder.forward(source, encoder_hidden, encoder_cell)
        if (self.encoder.cell_type == 'rnn'):
          encoder_outputs, encoder_hidden = self.encoder.forward(source, encoder_hidden, encoder_cell)
        if (self.encoder.cell_type == 'gru'):
          encoder_outputs, encoder_hidden = self.encoder.forward(source, encoder_hidden, encoder_cell)
           
        #first input to the decoder is the <sos> tokens
        input = target[:,0]
        self.sos = target[:,0]
        hidden,cell = self.decoder.init_hidden(encoder_hidden, encoder_cell, self.encoder.bidirectional)
        
        for t in range(1, target_len):
            output, hidden, cell,_ = self.decoder.forward(input, encoder_outputs, hidden, cell)
            outputs[:,t] = output.squeeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(-1)
            input = target[:,t] if teacher_force else top1.squeeze(1)

        return outputs

    def inference(self, source, target):
      
        batch_size = source.shape[0]
        target_len = self.max_target_length
        target_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)

        encoder_hidden, encoder_cell = self.encoder.init_hidden(batch_size)

        
        if (self.encoder.cell_type == 'lstm'):
          encoder_outputs, encoder_hidden, encoder_cell = self.encoder.forward(source, encoder_hidden, encoder_cell)
        if (self.encoder.cell_type == 'rnn'):
          encoder_outputs, encoder_hidden = self.encoder.forward(source, encoder_hidden, encoder_cell)
        if (self.encoder.cell_type == 'gru'):
          encoder_outputs, encoder_hidden = self.encoder.forward(source, encoder_hidden, encoder_cell)
           
        #first input to the decoder is the <sos> tokens
        input = self.sos
        input_len = encoder_outputs.shape[1]
        hidden,cell = self.decoder.init_hidden(encoder_hidden, encoder_cell, self.encoder.bidirectional)
        attention_map = torch.zeros(batch_size,target_len,input_len)
        
        for t in range(1, target_len):
            output, hidden, cell, attention = self.decoder.forward(input,encoder_outputs, hidden, cell)
            attention_map[:,t-1,:] = attention.squeeze(1)
            outputs[:,t] = output.squeeze(1)
            top1 = output.argmax(-1)
            input = top1.squeeze(1)
                
        return outputs, attention_map


def trainIters(model, pairs, batch_size, n_iters, optimizer, tf, print_every=10, plot_every=10, log = True, Attention = False):
    start = time.time()
    plot_losses = []
    train_char_accuracy = []
    train_word_accuracy = []
    val_losses = []
    val_char_accuracy = []
    val_word_accuracy = []
    print_loss_total = 0  
    plot_loss_total = 0  
    print_val_loss_total = 0
    plot_val_loss_total = 0 

    train_pairs = pairs[0]
    val_pairs = pairs[1]
    train_accuracy = 0
    
    criterion = nn.NLLLoss()

    count = 0
    for iter in range(1,n_iters+1): 
      for i in np.arange(start=0, stop=len(train_pairs)-batch_size, step=batch_size):
        train_accuracy = 0
        count += 1
        if (i + batch_size > len(train_pairs)):
          batch_size = len(train_pairs) - i + 1  
        input_tensor = []
        target_tensor = []
        
        for j in range(batch_size):
            input_tensor.append(train_pairs[i+j][0])
            target_tensor.append(train_pairs[i+j][1]) 
            
        input_tensor = torch.stack(input_tensor).squeeze(1).long()#.cuda()
        target_tensor = torch.stack(target_tensor).squeeze(1).long()#.cuda()
        

        optimizer.zero_grad()
        if (count < 4000):
          out = model(input_tensor, target_tensor, teacher_forcing_ratio=tf)
        else:
          out = model(input_tensor, target_tensor , teacher_forcing_ratio=0)
        
        out = torch.permute(out,[0,2,1])
        loss = criterion(out, target_tensor)        
        train_accuracy_word = word_level_accuracy(target_tensor, out)*batch_size
        train_accuracy = train_accuracy + train_accuracy_word
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        print_loss_total += loss
        plot_loss_total += loss

        if count % 10 == 0:
            val_input_tensor = []
            val_target_tensor = []

            for j in range(batch_size):
                val_input_tensor.append(val_pairs[j][0])
                val_target_tensor.append(val_pairs[j][1]) 

            val_input_tensor = torch.stack(val_input_tensor).squeeze(1).long()#.cuda()
            val_target_tensor = torch.stack(val_target_tensor).squeeze(1).long()#.cuda()
            if (Attention == True):
                val_out,_ = model.inference(val_input_tensor, val_target_tensor)
                val_out = val_out.permute(0,2,1)
            else:
                val_out = model.inference(val_input_tensor, val_target_tensor)
                val_out = val_out.permute(0,2,1)
            val_loss = criterion(val_out, val_target_tensor)
            val_loss_sampled = val_loss
       
        if count % 10 == 0:
            print_loss_avg = print_loss_total / 800
            print_loss_total = 0
            print('%s (%d %d%%) %.7f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
                                      

        if count % 10 == 0:
            plot_loss_avg = plot_loss_total / 800
            plot_losses.append(plot_loss_avg.detach())
            plot_loss_total = 0
        
    
    plot_losses = [losses.cpu().numpy() for losses in plot_losses]
    plot_losses_sampled = sample_equidistant_points(plot_losses, n_iters)#Need to log this list
    
    char_count = 0
    word_count = 0
        
    for i in np.arange(start=0, stop=len(val_pairs)-batch_size, step=batch_size):
        if (i + batch_size > len(val_pairs)):
          batch_size = len(val_pairs) - i + 1  
        val_input_tensor = []
        val_target_tensor = []
        for j in range(batch_size):
            val_input_tensor.append(val_pairs[i+j][0])
            val_target_tensor.append(val_pairs[i+j][1])

        val_input_tensor = torch.stack(val_input_tensor).squeeze(1).long()#.cuda()
        val_target_tensor = torch.stack(val_target_tensor).squeeze(1).long()#.cuda()
        if (Attention == True):
            val_out,_ = model.inference(val_input_tensor, val_target_tensor)
            val_out = val_out.permute(0,2,1)
        else:
            val_out = model.inference(val_input_tensor, val_target_tensor)
            val_out = val_out.permute(0,2,1)
        val_loss = criterion(val_out, val_target_tensor)
        val_accuracy_word = word_level_accuracy(val_target_tensor, val_out)
        word_count = word_count + (val_accuracy_word)*batch_size

    
    word_accuracy = word_count/(len(val_pairs)-batch_size) #Need to log this value = Val_Word_Accuracy
    
    metrics = {'Val_Accuracy': word_accuracy}                              
    print(f"Val loss = {val_loss}")
    print(f'Word-level-accuracy on val set = {word_accuracy}')


def append_strings_to_csv(file_path, string1, string2, string3):
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([string1, string2, string3])

def find_index(tensor, required_number):
    indices = np.where(tensor.cpu() == required_number)
    if indices[0].size > 0:
        return indices[0]+1
    else:
        return len(tensor)
    
def pad_idx(input_seq, target_seq, input_dict, target_dict):
    input_pad = input_dict[' ']
    target_pad = target_dict[' ']
    pad_idx_input = find_index(input_seq,input_pad)
    pad_idx_target = find_index(target_seq,target_pad)
    return pad_idx_input[0], pad_idx_target[0]

def test(model,test_pairs,criterion,file_path,input_dict,target_dict,Attention = False):
    test_word_count = 0
    for i in np.arange(start=0, stop=len(test_pairs)-32, step=32):
        batch_size = 32
        if (i + 32 > len(test_pairs)):
            batch_size = len(test_pairs) - i + 1  
        test_input_tensor = []
        test_target_tensor = []
        for j in range(batch_size):
            test_input_tensor.append(test_pairs[i+j][0])
            test_target_tensor.append(test_pairs[i+j][1])

        test_input_tensor = torch.stack(test_input_tensor).squeeze(1).long()#.cuda()
        test_target_tensor = torch.stack(test_target_tensor).squeeze(1).long()#.cuda()

        if (Attention == True):
            test_out, attention_map = model.inference(test_input_tensor, test_target_tensor)
        else :
            test_out = model.inference(test_input_tensor, test_target_tensor)
        test_out = torch.permute(test_out,[0,2,1])
        test_loss = criterion(test_out, test_target_tensor)
        for j in range(batch_size):
            input_str = print_keys_for_values(input_dict, test_input_tensor[j])
            output_str = print_keys_for_values(target_dict, torch.argmax(test_out[j], dim = 0))
            target_str = print_keys_for_values(target_dict, test_target_tensor[j])
            append_strings_to_csv(file_path, input_str, target_str, output_str)
        
        test_accuracy_word = word_level_accuracy(test_target_tensor, test_out)
        test_word_count = test_word_count + (test_accuracy_word)*batch_size

    test_word_accuracy = test_word_count/(len(test_pairs)-batch_size)
    print(f"Accuracy on Test Set is {test_word_accuracy}")

    if (Attention == True):
        for i in range(10):
            input_seq = test_input_tensor[i+10]
            target_seq = test_target_tensor[i+10]
            attention_seq = attention_map[i+10]
            pad_idx_input , pad_idx_target = pad_idx(input_seq, target_seq, input_dict, target_dict)
            # print(pad_idx_input , pad_idx_target)
            attention_seq_sub = attention_seq[0:int(pad_idx_input),0:int(pad_idx_target)]
            fig = plt.imshow(attention_seq_sub.detach().numpy() , cmap = 'gray')
            fig = fig.get_figure() 
            row_labels =  print_keys_for_values(input_dict, input_seq[0:pad_idx_input])
            column_labels = print_keys_for_values(target_dict, target_seq[0:pad_idx_target])
            plt.xticks(ticks=np.arange(len(column_labels)), labels=column_labels)
            plt.yticks(ticks=np.arange(len(row_labels)), labels=row_labels)
            plt.show()


def main(argv):
    input_dim = len(input_dict)
    output_dim = len(target_dict)
    batch_size = 32
    val_batch_size = 32
    enc_embedding = 256
    dec_embedding = 256
    hidden = 512
    enc_num_layers = 1
    dec_num_layers = 1
    enc_dropout = 0.4
    dec_dropout = 0.4
    max_length = max_target_length
    cell_type = 'lstm'
    Attention = True
    epochs = 5
    teacher_forcing_ratio = 0.7
    Bidirectional = True

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "i:e:d:k:l:h:c:g:a:p:t:b:" ,
                               ["enc_embed=","dec_embed=","enc_layers=","dec_layers=","hidden=",
                                "cell_type=","dropout=","attention=", "epochs=", "teacher_forcing_ratio=", "bidirectional=" ])

    for opt, arg in opts:
        if opt == '-i':
            print ('model.py --enc_embed <enc_embed> --dec_embed <dec_embed> --enc_layers <enc_layers> --dec_layers <dec_layers> --cell_type <cell_type> --dropout <dropout> --attention <Boolean- attention> --epochs <epochs> --teacher_forcing_ratio <tf> --bidirectional <Boolean-Bidirectional>')
            sys.exit()
        elif opt in ("-e", "--enc_embed"):
            enc_embedding = int(arg)
        elif opt in ("-d", "--dec_embed"):
            dec_embedding = int(arg)
        elif opt in ("-k", "--enc_layers"):
            enc_num_layers = int(arg)
        elif opt in ("-l", "--dec_layers"):
            dec_num_layers = int(arg)
        elif opt in ("-h", "--hidden"):
            hidden = int(arg)
        elif opt in ("-c", "--cell_type"):
            cell_type = arg
        elif opt in ("-g", "--dropout"):
            enc_dropout = float(arg)
            dec_dropout = float(arg)
        elif opt in ("-a", "--attention"):
            Attention = int(arg)
        elif opt in ("-p", "--epochs"):
            epochs = int(arg)
        elif opt in ("-t", "--teacher_forcing_ratio"):
            teacher_forcing_ratio = float(arg)
        elif opt in ("-b", "--bidirectional"):
            Bidirectional = int(arg)

    enc = EncoderRNN(device, cell_type, input_dim, enc_embedding, hidden, enc_num_layers, bidirectional = bool(Bidirectional), dropout_p = enc_dropout)
    if (bool(Attention) == True):
        dec = AttentionDecoderRNN(device, cell_type, output_dim, dec_embedding, hidden, max_length,  dec_dropout, dec_num_layers , bidirectional = bool(Bidirectional))
        model = AttentionSeq2Seq(enc, dec, device).to(device)
    else:
        dec = DecoderRNN(device, cell_type, output_dim, dec_embedding, hidden, max_length,  dec_dropout, dec_num_layers , bidirectional = bool(Bidirectional))
        model = Seq2Seq(enc, dec, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.NLLLoss()

    name = 'Predictions'
    file_path = name+'.csv'

    trainIters(model, pairs, 32, epochs, optimizer, teacher_forcing_ratio, Attention = bool(Attention))
    test(model,test_pairs,criterion,file_path,input_dict,target_dict, Attention = bool(Attention))

if __name__ == "__main__":
   main(sys.argv[1:])


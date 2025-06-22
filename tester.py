#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:43:34 2025

@author: Kshitij
"""
import os
try:
    os.chdir("Desktop/OMSCS/DL/assignment3_NLP_spr25")
except:
    pass
import numpy as np
import torch
from utils import *
from models.naive.LSTM import *
from models.seq2seq.Encoder import *
from models.seq2seq.Decoder import Decoder
from models.seq2seq.Seq2Seq import Seq2Seq
from models.Transformer import TransformerTranslator
import csv

set_seed_nb()
"""
i, n, h = 10, 2, 2

encoder = Encoder(i, n, h, h,model_type="LSTM")
x_array = np.random.rand(5,2) * 10
x = torch.LongTensor(x_array)
enc_out, enc_hidden = encoder.forward(x)

decoder = Decoder(h, n, n, i,model_type="LSTM")
x = torch.LongTensor(x_array[:,0]).unsqueeze(1) #decoder input is the first sequence
#enc_out, enc_hidden = unit_test_values('encoder')
out, hidden = decoder.forward(x,enc_hidden,enc_out)

embedding_size = 32
hidden_size = 32
input_size = 8
output_size = 8
batch, seq = 2, 2

encoder = Encoder(input_size, embedding_size, hidden_size, hidden_size,model_type="LSTM")
decoder = Decoder(embedding_size, hidden_size, hidden_size, output_size,model_type="LSTM")

seq2seq = Seq2Seq(encoder, decoder, 'cpu')
x_array = np.random.rand(batch, seq) * 10
x = torch.LongTensor(x_array)
out = seq2seq.forward(x)

# now lets test seq2seq with attention
decoder = Decoder(embedding_size, hidden_size, hidden_size, output_size, attention=True)
seq2seq = Seq2Seq(encoder, decoder, 'cpu')
out_attention = seq2seq.forward(x)

if out_attention is not None:
    expected_out = unit_test_values('seq2seq_attention')
    print('Close to out_attention: ', expected_out.allclose(out_attention, atol=1e-4))
else:
    print("SEQ2SEQ ATTENTION NOT IMPLEMENTED")

print(out_attention)
print(expected_out)

#print("Vocabulary Size:", len(word_to_ix))

#print(train_inxs.shape) # 7000 training instances, of (maximum/padded) length 43 words.
#print(val_inxs.shape) # 1551 validation instances, of (maximum/padded) length 43 words.
##print(train_labels.shape)
#print(val_labels.shape)

d1 = torch.load('./data/d1.pt')
d2 = torch.load('./data/d2.pt')
d3 = torch.load('./data/d3.pt')
d4 = torch.load('./data/d4.pt')

inputs = train_inxs[0:2]
inputs = torch.LongTensor(inputs)

model = TransformerTranslator(input_size=len(word_to_ix), output_size=2, device='cpu', hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=train_inxs.shape[1])

embeds = model.embed(inputs)

#print("Difference:", torch.sum(torch.pairwise_distance(embeds, d1)).item()) # should be very small (<0.01)
hidden_states = model.multi_head_attention(embeds)
#print("Difference:", torch.sum(torch.pairwise_distance(hidden_states, d2)).item()) # should be very small (<0.01)
outputs = model.feedforward_layer(hidden_states)
scores = model.final_layer(outputs)
"""
train_inxs = np.load('./data/train_inxs.npy')
val_inxs = np.load('./data/val_inxs.npy')
train_labels = np.load('./data/train_labels.npy')
val_labels = np.load('./data/val_labels.npy')

# load dictionary
word_to_ix = {}
with open("./data/word_to_ix.csv", "r", encoding='utf-8') as f:
    reader = csv.reader(f)
    for line in reader:
        word_to_ix[line[0]] = line[1]


from models.Transformer import FullTransformerTranslator
# you will be implementing and testing the forward function here. During training, inaddition to inputs, targets are also passed to the forward function
set_seed_nb()
inputs = train_inxs[0:3]
inputs[:,0]=0
inputs = torch.LongTensor(inputs)
inputs.to('cpu')
# Model
full_trans_model = FullTransformerTranslator(input_size=len(word_to_ix), output_size=5, device='cpu', hidden_dim=128, num_heads=2, dim_feedforward=2048, max_length=train_inxs.shape[1]).to('cpu')

tgt_array = np.random.rand(inputs.shape[0], inputs.shape[1])
targets = torch.LongTensor(tgt_array)
targets.to('cpu')
outputs = full_trans_model.forward(inputs,targets)
expected_out = unit_test_values('full_trans_fwd')
torch.sum(torch.pairwise_distance(outputs, expected_out)).item()
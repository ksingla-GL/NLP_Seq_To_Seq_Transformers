"""
S2S Decoder model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN", attention=False):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type
        self.attention = attention

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer based on the "model_type" argument.            #
        #          Supported types (strings): "RNN", "LSTM". Instantiate the        #
        #          appropriate layer for the specified model_type.                  #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #       5) If attention is True, A linear layer to downsize concatenation   #
        #           of context vector and input                                     #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
        self.embed1 = nn.Embedding(output_size,emb_size)
        if model_type == 'RNN':
            self.enc2 = nn.RNN(emb_size, decoder_hidden_size,batch_first=True)
        else:
            self.enc2 = nn.LSTM(emb_size, decoder_hidden_size,batch_first=True)
        self.fc3 = nn.Linear(decoder_hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.drop0 = nn.Dropout(dropout)
        if self.attention:
            self.fc2 = nn.Linear(decoder_hidden_size + emb_size, emb_size)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def compute_attention(self, hidden, encoder_outputs):
        """ compute attention probabilities given a controller state (hidden) and encoder_outputs using cosine similarity
            as your attention function.

                cosine similarity (q,K) =  q@K.Transpose / |q||K|
                hint |K| has dimensions: N, T
                Where N is batch size, T is sequence length

            Args:
                hidden (tensor): the controller state (dimensions: 1,N, hidden_dim)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention (dimensions: N,T, hidden dim)
            Returns:
                attention: attention probabilities (dimension: N,1,T)
        """

        #############################################################################
        #                              BEGIN YOUR CODE                              #
        # It is recommended that you implement the cosine similarity function from  #
        # the formula given in the docstring. This exercise will build up your     #
        # skills in implementing mathematical formulas working with tensors.        #
        # Alternatively you may use nn.torch.functional.cosine_similarity or        #
        # some other similar function for your implementation.                      #
        #############################################################################

        
        # Matmul with transpose on last dim (hidden_dim) -> (N, T)
        dot = torch.bmm(encoder_outputs, hidden.squeeze(0).unsqueeze(-1)).squeeze(-1)  #(N, T)

        q_norm = torch.norm(hidden.squeeze(0), dim=-1, keepdim=True)  #(N, 1)
        K_norm = torch.norm(encoder_outputs, dim=-1)  #(N, T)

        cosines = dot / (q_norm * K_norm + 1e-9)
        attention_prob = (nn.functional.softmax(cosines)).reshape(cosines.shape[0],1,cosines.shape[1])
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return attention_prob

    def forward(self, input, hidden, encoder_outputs=None):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden state of the previous time step from the decoder, dimensions: (1,N,decoder_hidden_size)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention, dimensions: (N,T,encoder_hidden_size)
                attention (Boolean): If True, need to implement attention functionality
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
                hidden (tensor): the state coming out of the hidden unit, dimensions: (1,N,decoder_hidden_size)
                where N is the batch size, T is the sequence length
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       1) Apply the dropout to the embedding layer                         #
        #                                                                           #
        #       2) If attention is true, compute the attention probabilities and    #
        #       use them to do a weighted sum on the encoder_outputs to determine   #
        #       the context vector. The context vector is then concatenated with    #
        #       the output of the dropout layer and is fed into the linear layer    #
        #       you created in the init section. The output of this layer is fed    #
        #       as input vector to your recurrent layer. Refer to the diagram       #
        #       provided in the Jupyter notebook for further clarifications. note   #
        #       that attention is only applied to the hidden state of LSTM.         #
        #                                                                           #
        #       3) Apply linear layer and log-softmax activation to output tensor   #
        #       before returning it.                                                #
        #                                                                           #
        #       If model_type is LSTM, the hidden variable returns a tuple          #
        #       containing both the hidden state and the cell state of the LSTM.    #
        #############################################################################

        #output, hidden = None, None     #remove this line when you start implementing your code
        out = self.embed1(input)
        out = self.drop0(out)
        #print(out.shape)
        if self.attention:
            n,t,h = encoder_outputs.shape
            if self.model_type == 'LSTM':
                probs = self.compute_attention(hidden[0], encoder_outputs)
            else:
                probs = self.compute_attention(hidden, encoder_outputs)
            #probs = probs.reshape(probs.shape[0],probs.shape[2],1)
            xyz = (encoder_outputs * probs.squeeze(1).unsqueeze(-1)).sum(dim = 1).unsqueeze(1)
            out = self.fc2(torch.concat((xyz,out),dim = -1))
                
        hid = self.enc2(out,hidden)
        if self.model_type == 'LSTM':
            hidden = hid[1]
            output = self.log_softmax(self.fc3(hidden[0][0]))
        else:
            hidden = hid[1]
            output = self.log_softmax(self.fc3(hidden[0])) 

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden

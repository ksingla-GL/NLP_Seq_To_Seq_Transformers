import random
import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model 
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        #############################################################################
        #    Following layers of the encoder are initialized:                       #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer based on the "model_type" argument.            #
        #          Supported types (strings): "RNN", "LSTM". Instantiate the        #
        #          appropriate layer for the specified model_type.                  #
        #       3) Linear layers with ReLU activation in between to get the         #
        #          hidden states of the Encoder(namely, Linear - ReLU - Linear).    #
        #          The size of the output of the first linear layer is the same as  #
        #          its input size.                                                  #
        #       4) A dropout layer                                                  #
        #                                                                           #
        #############################################################################
        self.embed1 = nn.Embedding(input_size,emb_size)
        if model_type == 'RNN':
            self.enc2 = nn.RNN(emb_size, encoder_hidden_size,batch_first=True)
        else:
            self.enc2 = nn.LSTM(emb_size, encoder_hidden_size,batch_first=True)
        self.fc3 = nn.Linear(encoder_hidden_size, encoder_hidden_size)
        self.fc4 = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.drop0 = nn.Dropout(dropout)
    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder;
                hidden (tensor): the state coming out of the last hidden unit
        """

        #############################################################################
        #       I will apply the dropout to the embedding layer before I apply the  #
        #       recurrent layer                                                     #
        #                                                                           #
        #       Apply tanh activation to the hidden tensor before returning it      #
        #                                                                           #
        #       Will not apply any linear layers/Relu for the cell state when       #
        #       model_type is LSTM before returning it.                             #
        #                                                                           #
        #       If model_type is LSTM, the hidden variable returns a tuple          #
        #       containing both the hidden state and the cell state of the LSTM.    #
        #############################################################################

        out = self.embed1(input)
        hidden = self.enc2(self.drop0(out))
        if self.model_type == 'LSTM':
            #print(hidden)
            #print(hidden.shape)
            output = hidden[0]
            h,c = hidden[1]
            h = nn.functional.tanh(self.fc4(nn.functional.relu(self.fc3(h))))
            hid = tuple([h,c])
            #hid[0] = nn.functional.tanh(self.fc4(nn.functional.relu(self.fc3(hid[0]))))
        else:
            hid = nn.functional.tanh(self.fc4(nn.functional.relu(self.fc3(hidden[1]))))
            output = hidden[0] 

        return output, hid

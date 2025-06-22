import random
import torch
import torch.nn as nn
import torch.optim as optim


# import custom models


class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
    """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        #############################################################################
        #    Initialize the Seq2Seq model.                                          #
        #############################################################################
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

    def forward(self, source):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
        """

        batch_size = source.shape[0]
        seq_len = source.shape[1]
        #############################################################################
        #       1) Will Get the last hidden representation from the encoder & use it#
        #          as the first hidden state of the decoder                         #
        #       2) The first input for the decoder should be the <sos> token, which #
        #          is the first in the source sequence.                             #
        #       3) Feed this first input and hidden state into the decoder          #  
        #          one step at a time in the sequence, adding the output to the     #
        #          final outputs.                                                   #
        #       4) Update the input and hidden being fed into the decoder           #
        #          at each time step. The decoder output at the previous time step  # 
        #          will have to be manipulated before being fed in as the decoder   #
        #          input at the next time step.                                     #
        #############################################################################

        #outputs = torch.zeros(batch_size,seq_len,self.decoder.output_size)
        # initially set outputs as a tensor of zeros with dimensions (batch_size, seq_len, decoder_output_size)
        enc_outs,hidden = self.encoder.forward(source)
        outputs = torch.tensor([])  #.reshape(batch_size,seq_len,self.decoder.output_size) 
        source = torch.LongTensor(source[:,0]).unsqueeze(1)
        for i in range(seq_len):
            out,hidden = self.decoder.forward(source,hidden,enc_outs)
            outputs = torch.concat((outputs,out.unsqueeze(1)),dim = 1)
            source = out.argmax(1).reshape(out.shape[0],1)
        return outputs

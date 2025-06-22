import numpy as np
import torch
from torch import nn
import random

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        
        ##############################################################################
        # Will Initialize what I need for the embedding lookup.                      #
        # I will need to use the max_length parameter above.                         #
        # Lets not worry about sine/cosine encodings- we use positional encodings.   #
        ##############################################################################
        self.embeddingL = nn.Embedding(input_size, self.word_embedding_dim)      #initialize word embedding layer
        self.posembeddingL =  nn.Embedding(max_length, self.word_embedding_dim)  #initialize positional embedding layer
                
        ##############################################################################
        # Initializations for multi-head self-attention.                             #
        ##############################################################################
        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        ##############################################################################
        # Will Initialize what I need for the feed-forward layer.                    # 
        # Don't forget the layer normalization.                                      #
        ##############################################################################
        
        self.l1 = nn.Linear(self.hidden_dim,self.dim_feedforward)
        self.l2 = nn.Linear(self.dim_feedforward,self.hidden_dim)

        ##############################################################################
        # Will Initialize what I need for the final layer.                           #
        ##############################################################################
        self.l3 = nn.Linear(self.hidden_dim,self.output_size)

    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        Will put together all of the layers I have developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.
        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """

        #############################################################################
        # Will Implement the full Transformer stack for the forward pass.           #
        # I will need to use all of the methods I have previously defined above.    #
        # I should only be calling TransformerTranslator class methods here.        #
        ############################################################################# 
        embeds = self.embed(inputs)
        hidden_states = self.multi_head_attention(embeds)
        outputs = self.feedforward_layer(hidden_states)
        outputs = self.final_layer(outputs)
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        #############################################################################
        # Note: word_to_ix has keys from 0 to self.vocab_size - 1                   #
        #############################################################################
      
        embeddings = None       #remove this line when you start implementing your code
        embeddings = self.embeddingL(inputs) + self.posembeddingL(torch.tensor([[i for i in range(inputs.shape[1])] \
                                                                                for j in range(inputs.shape[0])]))
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        
        #############################################################################
        # Now let's Implement multi-head self-attention followed by add + norm.     #
        #############################################################################
        q1 = self.q1(inputs)
        k1 = self.k1(inputs)
        head1 = torch.einsum('ijk,irk -> ijr',self.q1(inputs) , self.k1(inputs))
        head1 = head1 / np.sqrt(self.dim_q)
        #head1 = self.softmax(head1) @ self.v1(inputs)
        head1 = torch.einsum('ijk,ikl -> ijl',self.softmax(head1) ,self.v1(inputs))
        
        q2 = self.q2(inputs)
        k2 = self.k2(inputs)
        head2 = torch.einsum('ijk,irk -> ijr',self.q2(inputs) , self.k2(inputs))
        head2 = head2 / np.sqrt(self.dim_q)
        #head2 = self.softmax(head2) @ self.v2(inputs)
        head2 = torch.einsum('ijk,ikl -> ijl',self.softmax(head2) ,self.v2(inputs))
        outputs = self.attention_head_projection(torch.concat((head1,head2),dim = 2))
        #print(outputs.shape)
        #print(head1.shape)
        outputs = self.norm_mh(outputs + inputs)
        return outputs 
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # Now let's Implement the feedforward layer followed by add + norm.         #
        # Use a ReLU activation and apply the linear layers in the order we         #
        # initialized them.                                                         #
        #############################################################################
        outputs = self.norm_mh(self.l2(nn.functional.relu(self.l1(inputs))) + inputs)
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        #############################################################################
        # Now let's Implement the final layer for the Transformer Translator.       #
        # Softmax is not needed here as it is integrated as part of cross entropy   #
        # loss function.                                                            #
        #############################################################################
        outputs = self.l3(inputs)
        return outputs
        

class FullTransformerTranslator(nn.Module):

    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2,
                 dim_feedforward=2048, num_layers_enc=2, num_layers_dec=2, dropout=0.2, max_length=43, ignore_index=1):
        super(FullTransformerTranslator, self).__init__()

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.pad_idx=ignore_index

        seed_torch(0)

        ##############################################################################
        # Now let's Initialize what we need for the Transformer Layer                #
        ##############################################################################
        self.transformer = nn.Transformer(hidden_dim, num_heads, num_layers_enc, num_layers_dec,\
                                          dim_feedforward,dropout,batch_first = True)
        ##############################################################################
        # Now let's initialize what we need for the embedding lookup.                #
        # We will need to use the max_length parameter above.                        #
        # Initialize embeddings in order shown below.                                #
        # Let's not worry about sine/cosine encodings- weuse positional encodings.   #
        ##############################################################################
        self.srcembeddingL = nn.Embedding(self.input_size, hidden_dim)       #embedding for src
        self.tgtembeddingL = nn.Embedding(self.output_size, hidden_dim)       #embedding for target
        self.srcposembeddingL = nn.Embedding(max_length, hidden_dim)    #embedding for src positional encoding
        self.tgtposembeddingL = nn.Embedding(max_length, hidden_dim)    #embedding for target positional encoding
        ##############################################################################
        # Now let's initialize what we need for the final layer.                     #
        ##############################################################################
        self.final_layer = nn.Linear(hidden_dim, output_size)

    def forward(self, src, tgt):
        """
         This function computes the full Transformer forward pass used during training.
         Let's put together all of the layers we have developed in the correct order.

         :param src: a PyTorch tensor of shape (N,T) these are tokenized input sentences
                tgt: a PyTorch tensor of shape (N,T) these are tokenized translations
         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # Now let's implement the full Transformer stack for the forward pass.      #
        #############################################################################
        outputs=None
        # shift tgt to right, add one <sos> to the beginning and shift the other tokens to right
        tgt = self.add_start_token(tgt)


        # embed src and tgt for processing by transformer
        src_embeds = self.srcembeddingL(src) + self.srcposembeddingL(torch.tensor([[i for i in range(src.shape[1])] \
                                                    for j in range(src.shape[0])]))
        tgt_embeds = self.tgtembeddingL(tgt) + self.tgtposembeddingL(torch.tensor([[i for i in range(tgt.shape[1])] \
                                                    for j in range(tgt.shape[0])]))

        # create target mask and target key padding mask for decoder - Both have boolean values
        """ Generate target mask (upper triangular for causal attention)"""
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])        
        tgt_key_padding_mask = (tgt == self.pad_idx).to(self.device)
        
        src_key_padding_mask = (src == self.pad_idx).to(self.device)
        # invoke transformer to generate output
        transformer_out = self.transformer(
        src_embeds, tgt_embeds,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask)
        # pass through final layer to generate outputs
        outputs = self.final_layer(transformer_out)
        return outputs

    def generate_translation(self, src):
        """
         This function generates the output of the transformer taking src as its input
         it is assumed that the model is trained. The output would be the translation
         of the input

         :param src: a PyTorch tensor of shape (N,T)

         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # We will be calling the transformer forward function to generate the       #
        # translation for the input.                                                #
        #############################################################################
        tgt = torch.full((src.shape[0],src.shape[1]), self.pad_idx, dtype=torch.long, device=self.device)
        outputs = torch.zeros(src.shape[0],src.shape[1],self.output_size, device=self.device)
                
                           #used as an temporary variable to keep track of predicted tokens

        # initially set tgt as a tensor of <pad> tokens with dimensions (batch_size, seq_len) 
        tgt[:,0] = src[:,0]
        #tgt[:,1] = src[:,0]
        
        for t in range(src.shape[1]):
            # Run Transformer forward pass
            logits = self.forward(src, tgt[:,:t+1])  # Use current tgt sequence
            # Get predicted token for timestep t
            predicted_token = torch.argmax(logits[:, t, :], dim=-1)  # Last timestep output
            # Store logits & update tgt for next iteration
            outputs[:, t, :] = logits[:, t, :]
            if t+1 < self.max_length:
                tgt[:, t+1] = predicted_token
            #print("Pred token : ",predicted_token)
            #print("Logit : ",logits[:, t, :])
            #print("Should be : ",logits[:, t, :])
        return outputs

    def add_start_token(self, batch_sequences, start_token=2):
        """
            add start_token to the beginning of batch_sequence and shift other tokens to the right
            if batch_sequences starts with two consequtive <sos> tokens, return the original batch_sequence

            example1:
            batch_sequence = [[<sos>, 5,6,7]]
            returns:
                [[<sos>,<sos>, 5,6]]

            example2:
            batch_sequence = [[<sos>, <sos>, 5,6,7]]
            returns:
                [[<sos>, <sos>, 5,6,7]]
        """
        def has_consecutive_start_tokens(tensor, start_token):
            """
                return True if the tensor has two consecutive start tokens
            """
            consecutive_start_tokens = torch.tensor([start_token, start_token], dtype=tensor.dtype,
                                                    device=tensor.device)

            # Check if the first two tokens in each sequence are equal to consecutive start tokens
            is_consecutive_start_tokens = torch.all(tensor[:, :2] == consecutive_start_tokens, dim=1)

            # Return True if all sequences have two consecutive start tokens at the beginning
            return torch.all(is_consecutive_start_tokens).item()

        if has_consecutive_start_tokens(batch_sequences, start_token):
            return batch_sequences

        # Clone the input tensor to avoid modifying the original data
        modified_sequences = batch_sequences.clone()

        # Create a tensor with the start token and reshape it to match the shape of the input tensor
        start_token_tensor = torch.tensor(start_token, dtype=modified_sequences.dtype, device=modified_sequences.device)
        start_token_tensor = start_token_tensor.view(1, -1)

        # Shift the words to the right
        modified_sequences[:, 1:] = batch_sequences[:, :-1]

        # Add the start token to the first word in each sequence
        modified_sequences[:, 0] = start_token_tensor

        return modified_sequences

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

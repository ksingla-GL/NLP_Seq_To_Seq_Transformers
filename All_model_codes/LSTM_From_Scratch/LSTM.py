import numpy as np
import torch
import torch.nn as nn

class MyLSTMCell(torch.nn.Module):

  def __init__(self, input_size=10, hidden_size=64):
    super(MyLSTMCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.forgot_fc = nn.Linear(input_size,hidden_size)
    self.input_fc = nn.Linear(input_size,hidden_size)
    self.cell_fc = nn.Linear(input_size,hidden_size)
    self.output_fc = nn.Linear(input_size,hidden_size)

    self.forgot_fc_x = nn.Linear(input_size,hidden_size)
    self.input_fc_x = nn.Linear(input_size,hidden_size)
    self.cell_fc_x = nn.Linear(input_size,hidden_size)
    self.output_fc_x = nn.Linear(input_size,hidden_size)

  ### The Forget Gate takes in the input (x) and hidden state (h)
  ### The input and hidden state pass through their own linear compression layers,
  ### then are concatenated and passed through a sigmoid
  def forget_gate(self, x, h):
    f = None 
    f = self.forgot_fc_x(x) + self.forgot_fc(h)
    return f

  ### The Input Gate takes the input (x) and hidden state (h)
  ### The input and hidden state pass through their own linear compression layers,
  ### then are concatenated and passed through a sigmoid
  def input_gate(self, x, h):
    i = None 
    i = self.input_fc_x(x) + self.input_fc(h)
    return i

  ### The Cell memory gate takes the results from the input gate (i), the results from the forget gate (f)
  ### the original input (x), the hidden state(h) and the previous cell state (c_prev).
  ### 1. The Cell memory gate compresses the input and hidden and concatenates them and passes it through a Tanh.
  ### 2. The resultant intermediate tensor is multiplied by the results from the input gate to determine
  ###    what new information is allowed to carry on
  ### 3. The results from the forget state are multiplied against the previous cell state (c_prev) to determine
  ###    what should be removed from the cell state.
  ### 4. The new cell state (c_next) is the new information that survived the input gate and the previous
  ###    cell state that survived the forget gate.
  ### The new cell state c_next is returned
  def cell_memory(self, i, f, x, h, c_prev):
    c_next = None
    c_next = nn.functional.tanh(self.cell_fc_x(x) + self.cell_fc(h)) * i + c_prev * f
    return c_next

  ### The Out gate takes the original input (x) and the hidden state (h)
  ### The gate passes the input and hidden through their own compression layers and
  ### then concatenates to send through a sigmoid
  def out_gate(self, x, h):
    o = None 
    o = self.output_fc_x(x) + self.output_fc(h)
    return o

  ### This function assembles the new hidden state, give the results of the output gate (o)
  ### and the new cells sate (c_next).
  ### This function runs c_next through a tanh to get a 1 or -1 which will flip some of the
  ### elements of the output.
  def hidden_out(self, o, c_next):
    h_next = None
    h_next = o * nn.functional.tanh(c_next)
    return h_next

  def forward(self, x, hc):
    (h, c_prev) = hc
    # Equation 1. input gate
    i = self.input_gate(x, h)

    # Equation 2. forget gate
    f = self.forget_gate(x, h)

    # Equation 3. updating the cell memory
    c_next = self.cell_memory(i, f, x, h, c_prev)

    # Equation 4. calculate the main output gate
    o = self.out_gate(x, h)

    # Equation 5. produce next hidden output
    h_next = self.hidden_out(o, c_next)

    return h_next, c_next

  def init_hidden(self):
    return (torch.zeros(1, self.hidden_size),
            torch.zeros(1, self.hidden_size))


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        #   LSTM weights and attributes are declared in order specified below.         #
        #   I should include weights and biases regarding using nn.Parameter:          #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   for each equation above, initialize the weights,biases for input prior     #
        #   to weights, biases for hidden.                                             #
        #   when initializing the weights consider that in forward method I          #
        #   should NOT transpose the weights.                                          #
        #   I also need to include correct activation functions                        #
        ################################################################################

        # i_t: input gate
        self.x_input_weights = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.x_input_bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.w_input_weights = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))                                 
        self.w_input_bias = nn.Parameter(torch.zeros(self.hidden_size))
        # f_t: the forget gate
        self.x_forget_weights = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.x_forget_bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.w_forget_weights = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))                                 
        self.w_forget_bias = nn.Parameter(torch.zeros(self.hidden_size))
        # g_t: the cell gate
        self.x_cell_weights = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.x_cell_bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.w_cell_weights = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))                                 
        self.w_cell_bias = nn.Parameter(torch.zeros(self.hidden_size))
        # o_t: the output gate
        self.x_output_weights = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.x_output_bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.w_output_weights = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))                                 
        self.w_output_bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              #
        #   h_t and c_t should be initialized to zeros.                                #
        #   Note that this time I am also iterating over all of the time steps.     #
        ################################################################################
        h,c = torch.zeros(x.shape[0],self.hidden_size), torch.zeros(x.shape[0],self.hidden_size)  
        #x = torch.concat((x,h),dim = 2)
        for i in range(x.shape[1]):
            x_t = x[:,i,:]
            input_gate = nn.functional.sigmoid(x_t @ self.x_input_weights + \
                            h @ self.w_input_weights + self.x_input_bias + self.w_input_bias)
            forget_gate = nn.functional.sigmoid(x_t @ self.x_forget_weights + \
                            h @ self.w_forget_weights + self.x_forget_bias + self.w_forget_bias)
            cell_gate = nn.functional.tanh(x_t @ self.x_cell_weights + \
                            h @ self.w_cell_weights + self.x_cell_bias + self.w_cell_bias)
            output_gate = nn.functional.sigmoid(x_t @ self.x_output_weights + \
                            h @ self.w_output_weights + self.x_output_bias + self.w_output_bias)
            
            c = c * forget_gate + cell_gate * input_gate
            h = output_gate * nn.functional.tanh(c)
        return (h,c)

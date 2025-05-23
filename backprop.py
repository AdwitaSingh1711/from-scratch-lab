import torch
import torch.nn as nn

class randomNet():
  def __init__(self, input_size, hidden_size, output_size):

    # number of neurons per layer
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    # initialise 1-2 weight cuz me bored
    self.w1 = torch.randn(self.input_size, self.hidden_size)
    self.w2 = torch.randn(self.hidden_size, self.output_size)

  def forward(self, x):
    """
    x: (batch_size, input_size)
    """
    # n1 = nn.Linear(input_size, hidden_size)
    n1 = x @ self.w1
    n2 = torch.tanh(n1)
    n3 = n2 @ self.w2
    
    return n1, n2, n3
  
  def loss(output, target):
    return (output-target)**2
  
  def backward(self, x, n1, n2, n3, output, target):
    """
    x: (batch_size, input_size)
    n1: (batch_size, hidden_size)
    n2: (batch_size, hidden_size)
    n3: (batch_size, output_size)
    """

    dn3 = 2*(output-target)
    dw2 = n2.T @ dn3
    dn2 = dn3 @ self.w2.T
    dtanh = (1 - torch.tanh(dw2)**2)
    dn1 = dn2 * dtanh
    dw1 = x.T @ dn1

    return dw2, dw1

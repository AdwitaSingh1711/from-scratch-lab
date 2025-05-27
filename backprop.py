import torch
import torch.nn as nn

class RandomeNet:
    def __init__(self):
        self.w1 = torch.randn(4, 6)   # input, hidden
        self.w2 = torch.randn(6, 5)   # hidden, hidden
        self.w3 = torch.randn(5, 2)   # hidden → output

    def forward(self, x):
        z1 = x @ self.w1             
        a1 = torch.relu(z1)          

        z2 = a1 @ self.w2             
        a2 = torch.tanh(z2)          

        z3 = a2 @ self.w3            
        return z1, a1, z2, a2, z3    

    def loss(z3, target):
        return (z3-target)**2 
    
    def backward2(self, x, z1, a1, z2, a2, z3, target):
        dz3 = 2 * (z3 - target)           # (batch_size, 2)
        dw3 = a2.T @ dz3                  # (5, 2)

        da2 = dz3 @ self.w3.T            # (batch_size, 5)
        dtanh = 1 - torch.tanh(z2) ** 2  # (batch_size, 5)
        dz2 = da2 * dtanh                # (batch_size, 5)
        dw2 = a1.T @ dz2                 # (6, 5)

        da1 = dz2 @ self.w2.T            # (batch_size, 6)
        drelu = (z1 > 0).float()         # (batch_size, 6)
        dz1 = da1 * drelu                # (batch_size, 6)
        dw1 = x.T @ dz1                  # (4, 6)

        return dw1, dw2, dw3



# class randomNet():
#   def __init__(self, input_size, hidden_size, output_size):

#     # number of neurons per layer
#     self.input_size = input_size
#     self.hidden_size = hidden_size
#     self.output_size = output_size

#     # initialise 1-2 weight cuz me bored
#     self.w1 = torch.randn(self.input_size, self.hidden_size)
#     self.w2 = torch.randn(self.hidden_size, self.output_size)

#   def forward(self, x):
#     """
#     x: (batch_size, input_size)
#     """
#     # n1 = nn.Linear(input_size, hidden_size)
#     n1 = x @ self.w1
#     n2 = torch.tanh(n1)
#     n3 = n2 @ self.w2
    
#     return n1, n2, n3
  
#   def loss(output, target):
#     return (output-target)**2
  
#   def backward(self, x, n1, n2, n3, output, target):
#     """
#     x: (batch_size, input_size)
#     n1: (batch_size, hidden_size)
#     n2: (batch_size, hidden_size)
#     n3: (batch_size, output_size)
#     """

#     dn3 = 2*(output-target)
#     dw2 = n2.T @ dn3
#     dn2 = dn3 @ self.w2.T
#     dtanh = (1 - torch.tanh(dw2)**2)
#     dn1 = dn2 * dtanh
#     dw1 = x.T @ dn1

#     return dw2, dw1
  

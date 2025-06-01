import torch
import torch.nn as nn

class RandomeNet:
    def __init__(self):
        self.w1 = torch.randn(4, 6)   # input, hidden
        self.w2 = torch.randn(6, 5)   # hidden, hidden
        self.w3 = torch.randn(5, 2)   # hidden → output
        self.params = [self.w1, self.w2, self.w3]

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
    
    def adam_optimizer(self, grad, max_steps = 1000, alpha = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
      # m0 = torch.zeros(params.shape)
      # v0 = torch.zeros(params.shape)
      m0 = [torch.zeros_like(p) for p in self.params]
      v0 = [torch.zeros_like(p) for p in self.params]
      # mt = [m0]
      # vt = [v0]

      for t in range(1,max_steps+1):
        for i in range(len(self.params)):
          gt = grad[i];
          # mt = beta_1 * mt[-1] + (1-beta_1) * gt
          mt[i] = beta_1 * mt[i] + (1-beta_1) * gt
          # vt = beta_2 * vt[-1] + (1-beta_2) * gt**2
          vt[i] = beta_2 * vt[i] + (1-beta_2) * gt**2
          # m_1.append(mt)
          # v_1.append(vt)

          m_hat = mt[i] / (1-beta_1**t)
          v_hat = vt[i]/ (1-beta_2**t)

          self.params[i] -= - alpha * m_hat / (v_hat**0.5 + epsilon)

      
      return params


def literal_math_of_adam_optimizer(self, grad, max_steps = 1000, alpha = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 10**(-8)):
      self.params = params # list of tensorsjkjjkk
      mt=[[torch.zeros_like(p) for p in params]]
      vt=[[torch.zeros_like(p) for p in params]]

      m_curr = []
      v_curr = []


      for t in range(max_steps):
        for i in range(len(params)):
          g = grad[i]

          m_t = beta_1 * mt[-1] + (1-beta_1) * g
          v_t = beta_2 * vt[-1] + (1-beta_2) * g**2

          m_hat = m_t / (1-beta_1**t)
          v_hat = v_t/ (1-beta_2**t)

          params[i] -= alpha * m_hat / (v_hat**0.5 + epsilon)

          m_curr.append(m_t)
          v_curr.append(v_t)

        mt.append(m_curr)
        vt.append(v_curr)

      
      return params



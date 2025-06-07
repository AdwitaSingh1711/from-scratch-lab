import torch
import torch.nn as nn

class RandomeNet:
    def __init__(self):
        self.w1 = torch.randn(4, 6)   # input, hidden
        self.w2 = torch.randn(6, 5)   # hidden, hidden
        self.w3 = torch.randn(5, 2)   # hidden → output
        self.params = [self.w1, self.w2, self.w3]

        self.mt = [torch.zeros_like(p) for p in self.params]
        self.vt = [torch.zeros_like(p) for p in self.params]
        self.t = 0 # timestep for adam

    def forward(self, x):
        z1 = x @ self.w1             
        a1 = torch.relu(z1)          

        z2 = a1 @ self.w2             
        a2 = torch.tanh(z2)          

        z3 = a2 @ self.w3            
        return z1, a1, z2, a2, z3    

    def loss(self, z3, target):
        return torch.mean((z3-target)**2) 
    
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

      """
      grad: gradients calculated during the backward pass [dw1, dw2, dw3]
      
      """
      # m0 = torch.zeros(params.shape)
      # v0 = torch.zeros(params.shape)
      self.t+=1

      # mt = [m0]
      # vt = [v0]

      for i in range(len(self.params)):
        gt = grad[i];
        # mt = beta_1 * mt[-1] + (1-beta_1) * gt
        self.mt[i] = beta_1 * self.mt[i] + (1-beta_1) * gt
        # vt = beta_2 * vt[-1] + (1-beta_2) * gt**2
        self.vt[i] = beta_2 * self.vt[i] + (1-beta_2) * gt**2
        # m_1.append(mt)
        # v_1.append(vt)

        m_hat = self.mt[i] / (1-beta_1**self.t)
        v_hat = self.vt[i]/ (1-beta_2**self.t)

        self.params[i] -= - alpha * m_hat / (v_hat**0.5 + epsilon)

    
    def train_with_adam(self, X, y, epochs=1000, alpha=0.001):
        """
        Training loop that implements the "while θt has not converged do" concept practically
        We use epochs (max_steps) as the convergence criterion
        """
        losses = []
        
        for epoch in range(epochs):  # This is your "max_steps" concept!
            # Forward pass
            z1, a1, z2, a2, z3 = self.forward(X)
            
            # Compute loss
            loss_val = self.loss(z3, y)
            losses.append(loss_val.item())
            
            # Backward pass - compute fresh gradients
            gradients = self.backward2(X, z1, a1, z2, a2, z3, y)
            
            # Single Adam step with current gradients
            self.adam_step(gradients, alpha)
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_val.item():.6f}")
        
        return losses
    

if __name__ == "__main__":
    print("-"*20)
    print("\nRandomNet working with Adam\n")
    print("-"*20)

    torch.manual_seed(42)
    X = torch.randn(100, 4)
    y = torch.randn(100, 2)
    
    print("\n" + "="*60)
    print("TESTING CORRECTED VERSION")
    print("="*60)
    
    model = RandomeNet_Corrected()
    losses = model.train_with_adam(X, y, epochs=500, alpha=0.01)
    
    print(f"\nFinal loss: {losses[-1]:.6f}")
    print(f"Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")



# def literal_math_of_adam_optimizer(self, grad, max_steps = 1000, alpha = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 10**(-8)):
#       self.params = params # list of tensorsjkjjkk
#       mt=[[torch.zeros_like(p) for p in params]]
#       vt=[[torch.zeros_like(p) for p in params]]

#       m_curr = []
#       v_curr = []


#       for t in range(max_steps):
#         for i in range(len(params)):
#           g = grad[i]

#           mt = beta_1 * mt[-1] + (1-beta_1) * g
#           vt = beta_2 * vt[-1] + (1-beta_2) * g**2

#           m_hat = mt / (1-beta_1**t)
#           v_hat = vt/ (1-beta_2**t)

#           params[i] -= alpha * m_hat / (v_hat**0.5 + epsilon)

#           m_curr.append(mt)
#           v_curr.append(vt)

#         mt.append(m_curr)
#         vt.append(v_curr)

      
#       return params



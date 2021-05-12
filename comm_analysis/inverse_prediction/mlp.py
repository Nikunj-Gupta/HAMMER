import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim, numpy as np 
device = torch.device("cpu")

class InverseModel(nn.Module):
    """ 
    We receive messages from the evaluation model of HAMMER.
    We also receive observations from a numpy array. (serving as ground truth)
    """ 

    def __init__(self, in_dim: int, hidden_layers, out_dim: int):
        """Initialization."""
        super(InverseModel, self).__init__()

        l_dim = [in_dim] 
        l_dim.extend(hidden_layers) 

        layers = [] 
        for l in range(len(l_dim) - 1): 
            layers.append(nn.Linear(l_dim[l], l_dim[l+1]) )
            layers.append(nn.ReLU())        
        layers.append(nn.Linear(l_dim[-1], out_dim)) 
        self.model = nn.Sequential(*layers) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.model(x) 


if __name__ == '__main__':

    model = InverseModel(in_dim=1, hidden_layers=[64, 64], out_dim=54)
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 

    """ 
    X = (import messages)
    Y = (import actual observations)
    """

    X = torch.rand(500, 1, 1) 
    Y = torch.rand(500, 1, 54)

    n_epochs = 10
    batch_size = 32

    for epoch in range(n_epochs):
        permutation = torch.randperm(X.size()[0])

        for i in range(0, X.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X[indices], Y[indices]

            outputs = model.forward(batch_x)
            loss = F.mse_loss(outputs, batch_y)

            loss.backward()
            optimizer.step() 
            

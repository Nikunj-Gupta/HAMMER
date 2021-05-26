import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim, numpy as np, argparse, os 
device = torch.device("cpu")

class VAE(nn.Module):
    """ 
    We receive messages from the evaluation model of HAMMER.
    We also receive observations from a numpy array. (serving as ground truth)
    """ 

    def __init__(self, in_dim: int, hidden_layers, out_dim: int):
        """Initialization."""
        super(VAE, self).__init__()

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
    parser = argparse.ArgumentParser() 
    parser.add_argument("--filepath", type=str) 
    parser.add_argument("--train", type=int, default=1) 
    parser.add_argument("--save", type=int, default=1) 
    parser.add_argument("--load", type=int, default=1) 
    parser.add_argument("--epochs", type=int, default=100) 
    parser.add_argument("--batch_size", type=int, default=512) 

    args = parser.parse_args() 

    where = args.filepath + "/inverse" 
    if not os.path.exists(where): os.makedirs(where) 


    """ 
    X = (import messages)
    Y = (import actual observations)
    """
    X = np.load(os.path.join(args.filepath, "hammer_states.npy"), mmap_mode="r") 
    X = np.array([i.reshape(1, -1) for i in X]) 
    X = torch.from_numpy(X) 

    Y = np.load(os.path.join(args.filepath, "hammer_states.npy"), mmap_mode="r") 
    Y = np.array([i.reshape(1, -1) for i in Y]) 
    Y = torch.from_numpy(Y) 

    model = VAE(in_dim=X.shape[-1], hidden_layers=[64, 64, 1, 64, 64], out_dim=Y.shape[-1]) # For VAE 

    n_epochs = args.epochs 
    batch_size = args.batch_size 

    bar = int((90/100) * X.shape[0])  # training and testing set 
    print(bar)

    if args.train: 
        X_train = X[:bar] 
        Y_train = Y[:bar] 
        print("X:", X_train.shape) 
        print("Y: ", Y_train.shape)     

        optimizer = optim.Adam(model.parameters(), lr=1e-3) 

        for epoch in range(n_epochs):
            losses = []
            permutation = torch.randperm(X_train.size()[0])

            for i in range(0, X_train.size()[0], batch_size):
                optimizer.zero_grad()

                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_train[indices], Y_train[indices]

                outputs = model.forward(batch_x)
                loss = F.mse_loss(outputs, batch_y) 
                losses.append(loss.detach().item()) 

                loss.backward()
                optimizer.step() 
            print("Epoch: ", epoch, " Mean Loss: ", np.mean(losses)) 
    
        if args.save: 
            torch.save(model.state_dict(), os.path.join(where, "inverse_model.dict")) 

    if args.load: 
        model.load_state_dict(torch.load(os.path.join(where, "inverse_model.dict"))) 
        model.eval() 
        X_test = X[bar:] 
        Y_test = Y[bar:] 
        print("X:", X_test.shape) 
        print("Y: ", Y_test.shape)     

        outputs = model.forward(X_test) 
        loss = F.mse_loss(outputs, Y_test) 
        print("Test Mean Loss: ", loss.detach().item()) 

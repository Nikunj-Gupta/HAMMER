from threading import local
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 

local_states =np.load("./save\dataset\local_states.npy", allow_pickle=True) 
num_bins = 200
X_embedded = TSNE(n_components=2).fit_transform(local_states[:,1,:])
for i in range(19):
    gradients = local_states[:, 0, -1-i]
    print(i, max(gradients), min(gradients))
    # n, bins, patches = plt.hist(gradients, num_bins, facecolor='blue', alpha=0.5) 
    # plt.xlim((-1e-6,1e-6))

    sc = plt.scatter(X_embedded[:, 0],X_embedded[:, 1], c=gradients, cmap='RdYlBu') 
    plt.colorbar(sc)  
    plt.show()




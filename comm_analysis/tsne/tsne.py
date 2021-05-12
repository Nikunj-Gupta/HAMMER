import numpy as np 
from sklearn.manifold import TSNE 

X = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]]) 
print(X.shape)
X_embedded = TSNE(n_components=2).fit_transform(X) 
print(X_embedded.shape) 

print(X_embedded) 
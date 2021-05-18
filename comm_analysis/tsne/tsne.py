import numpy as np, argparse 
from sklearn.manifold import TSNE 

def tsne(filename): 
    X = np.load(filename, mmap_mode="r") 
    print(X.shape) 

    X_embedded = TSNE(n_components=2).fit_transform(X) 
    print(X_embedded.shape) 
    np.save("/".join(filename.split('/')[:-1])+"/embedded_states.npy", X_embedded) 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--filename", type=str, help="eval states") 
    args = parser.parse_args() 

    tsne(filename=args.filename) 


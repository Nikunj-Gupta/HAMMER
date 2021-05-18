import matplotlib.pyplot as plt, numpy as np, os, argparse 


def plot(embedded_filename, messages_filename, save=False): 
    X_embedded = np.load(embedded_filename) 
    print(X_embedded.shape) 
    X_embedded = X_embedded[:50000]
    print(X_embedded.shape) 
    messages = np.load(messages_filename) 

    x=[i[0] for i in X_embedded]
    y=[i[1] for i in X_embedded]
    z=[i[0] for i in messages[:len(X_embedded)]] 
    sc = plt.scatter(x,y, c=z, vmin=0., vmax=1., cmap='RdYlBu') 
    plt.colorbar(sc) 

    
    where = "/".join(embedded_filename.split('/')[:-1])+"/plots" 
    if not os.path.exists(where): os.makedirs(where) 
    if save: plt.savefig(os.path.join(where, "all.png")) 
    else: plt.show() 
    plt.close() 

    for j in range(25):     
        x=[X_embedded[i][0] for i in range(j, len(X_embedded), 25)] 
        y=[X_embedded[i][1] for i in range(j, len(X_embedded), 25)] 
        z=[messages[i][0] for i in range(j, len(X_embedded), 25)] 
        # print(len(x), len(y), len(z))
        sc = plt.scatter(x,y, c=z, vmin=0., vmax=1., cmap='RdYlBu') 
        plt.colorbar(sc) 
        if save: plt.savefig(os.path.join(where, "timestep_"+str(j+1)+".png")) 
        else: plt.show() 
        plt.close() 
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--filepath", type=str, help="eval states and messages (path)") 
    args = parser.parse_args() 
    plot(
        embedded_filename=os.path.join(args.filepath, "embedded_states.npy"),
        messages_filename=os.path.join(args.filepath, "hammer_messages.npy"),
        save=True 
    ) 

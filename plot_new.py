import os, glob, yaml, pprint, numpy as np, pandas as pd
from webbrowser import get 
from collections import defaultdict 
import matplotlib.pyplot as plt 
from tensorboard.backend.event_processing import event_accumulator
# from pylab import rcParams, axes



SIZE_GUIDANCE = {
    'compressedHistograms': 500, 
    'images': 4, 
    'audio': 4, 
    'scalars': 10000, 
    'histograms': 1, 
}

STORE_EVERYTHING_SIZE_GUIDANCE = {
    'compressedHistograms': 0, 
    'images': 0, 
    'audio': 0, 
    'scalars': 0, 
    'histograms': 0, 
} 
# params = {
#         'axes.labelsize': 28, 
#         'axes.titlesize': 32, 
#         'legend.fontsize': 14,
#         'xtick.labelsize': 'x-large',
#         'ytick.labelsize': 'x-large',
#         'text.usetex': True, 
#         'figure.figsize': [14, 12]
#     }

SMOOTH = 1000
CUT = 30000
def get_values(filename, scalar="Episodic_Reward"): 
    ea = event_accumulator.EventAccumulator(filename, size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
    ea.Reload()
    ea_scalar = ea.Scalars(tag=scalar) 
    ea_scalar = pd.DataFrame(ea_scalar) 
    return ea_scalar 


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



def save_numpy(log_dir): 
    logs = glob.glob(os.path.join(log_dir, "*/logs/*"), recursive=True) 
    l = [] 
    for event in logs: 
        l.append(event.split('/')[2].split('--')) 
    merged = defaultdict(lambda: []) 
    for i in l: merged['--'.join(i[:-1])].append('--'.join(i))
    count = -1 
    for exp in merged: 
        print("================================\n"+exp+"\n================================\n")
        vals = [] 
        count+=1
        for i in merged[exp]: 
            logs = glob.glob(os.path.join('./save', i, "logs/event*"), recursive=True)[0] 
            # save numpy arrays 
            logs_dir = glob.glob(os.path.join('./save', i, "logs"), recursive=True)[0] 
            with open(logs_dir+'/arr.npy', 'wb') as f: 
                np.save(f, get_values(logs)['value'].to_numpy()) 


def preprocess(log_dir): 
    
    params = {
        'axes.labelsize': 28, 
        'axes.titlesize': 32, 
        'legend.fontsize': 14,
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large',
        'text.usetex': True, 
        'figure.figsize': [10, 8]
    }
    from pylab import plot, rcParams, legend, axes, grid

    rcParams.update(params)
    
    
    logs = glob.glob(os.path.join(log_dir, "*/logs/*"), recursive=True) 

    # pprint.pprint(logs) 
    l = [] 
    for event in logs: 
        l.append(event.split('/')[2].split('--')) 
    # pprint.pprint(l) 
    merged = defaultdict(lambda: []) 
    for i in l: merged['--'.join(i[:-1])].append('--'.join(i))
    # pprint.pprint(merged) 
    # pprint.pprint(list(merged.keys()))



    count = -1 
    colors = ['#006BB2', '#B22400', '#006BB2', '#B22400', '#006BB2', '#B22400'] 
    labels = ['HAMMER', 'IL','HAMMER', 'IL','HAMMER', 'IL','HAMMER', 'IL'] 
    for exp in merged: 
        if ('n_3' in exp) and (('meslen_2' in exp) or ('meslen_0' in exp)): 
        # if ('n_3' in exp) and ('dru_0' in exp): 
            print("================================\n"+exp+"\n================================\n")
            vals = [] 
            count+=1
            for i in merged[exp]: 
                    # if ('randomseed_5327' in i): 
                    # if any(n in i for n in ['5327', '8651', '14712', '186', 
                    #                         '9538', '106', '90300', '4973', '310', '530', '606']): 
                    # if any(n in i for n in ['5327', '8651', '14712', '186', 
                    #                         '9538', '106']): 
                    print(i)
                    logs = glob.glob(os.path.join('./save', i, "logs/*.npy"), recursive=True)[0]                     
                    arr = np.load(logs) 
                    vals.append(smooth(arr[:CUT], box_pts=SMOOTH)[SMOOTH:CUT-SMOOTH]) 
                    # break 
            val_means = np.array(vals).mean(axis=0)
            val_stds = np.array(vals).std(axis=0)
            plt.plot(val_means, label=exp)
            # plt.plot(val_means, label=labels[count], color=colors[count])
            plt.fill_between(np.arange(1, val_means.shape[0]+1), 
                        val_means - val_stds, 
                        val_means + val_stds, 
                        alpha=0.1) 
            # break 

    rcParams.update(params)
    legend = plt.legend(loc="lower right")  
    # legend = legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True) 
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9') 

    plt.title('HAMMER') 
    plt.xlabel("Number of Episodes")
    plt.ylabel("Average Returns per Agent")
    plt.xlim((0, CUT-SMOOTH)) 
    plt.grid()
    plt.show() 
        


if __name__ == '__main__': 
    # save_numpy('./save') 
    preprocess('./save')

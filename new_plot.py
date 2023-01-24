import os, glob, numpy as np, matplotlib.pyplot as plt 
from tensorboard.backend.event_processing import event_accumulator

LOG_DIRS = ["./logs", "./save_cn_final", "./save_hammer_v1"] 
CUT = 30000 
SMOOTH = 100 
N_AGENTS = 5  


logs = glob.glob(os.path.join(LOG_DIRS[0], "*/**/*.npy"), recursive=True) 
for i in range(1, len(LOG_DIRS)): 
    logs2 = glob.glob(os.path.join(LOG_DIRS[i], "*/**/*.npy"), recursive=True) 
    logs.extend(logs2) 

# logs.sort() 
# print((len(logs))) 
# print(logs[:2]) 
# exit() 


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    # return np.array(pd.Series(y).rolling(box_pts).mean()) 
    return y_smooth


def get_label(exp_name): 
    if "hammer--cn--coil" in exp_name: 
        return "COIL" 
    if "dru_0--meslen_0" in exp_name: 
        return "IL" 
    if "dru_0--meslen_1" in exp_name: 
        return "HAMMERv2" 
    if "dru_1--meslen_1" in exp_name: 
        return "HAMMERv3" 
    if "hammer--cn--hammer-v1" in exp_name: 
        return "HAMMERv1" 
    if "hammer--cn--randommes" in exp_name: 
        return "IL + Random messages" 
    

seeds = set() 
for log in logs: 
    try: 
        seeds.add(int(log.split('/')[2].split('--')[-1].split('_')[-1])) 
    except: 
        seeds.add(int(log.split('/')[2].split('--')[-1][2:]))
# print(seeds) 

exps = set() 
for log in logs: 
    exps.add("--".join(log.split('/')[2].split('--')[:-1]))
# print(exps) 

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

for exp in exps: 
    # print(exp)

    if (("n_"+str(N_AGENTS) in exp)\
        and (("--meslen_0" in exp) or ("--meslen_1" in exp))\
        and not (("--dru_1" in exp) and ("--meslen_0" in exp)))\
        or (("n"+str(N_AGENTS) in exp) and ("--coil" in exp)) \
        or (("n"+str(N_AGENTS) in exp) and ("--randommes" in exp))\
        or (("n"+str(N_AGENTS) in exp) and ("--hammer-v1" in exp))\
            :        
        print(exp) 
        val_arr = [] 
        for log in logs: 
            if exp in log: 
                vals = np.load(log) 
                val_arr.append(vals[:CUT]) 

        val_means = np.array(val_arr).mean(axis=0)
        val_stds = np.array(val_arr).std(axis=0)

        if SMOOTH: 
            val_means = smooth(val_means, SMOOTH) 
            val_stds = smooth(val_stds, SMOOTH) 
            val_means = val_means[SMOOTH:-SMOOTH] 
            val_stds = val_stds[SMOOTH:-SMOOTH]

        plt.plot(val_means, label=get_label(exp)) 
        plt.fill_between(np.arange(1, val_means.shape[0]+1), 
                    val_means - val_stds, 
                    val_means + val_stds, 
                    alpha=0.1) 
rcParams.update(params)
legend = plt.legend(loc="lower right")  
# legend = legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True) 
frame = legend.get_frame()
frame.set_facecolor('0.9')
frame.set_edgecolor('0.9') 

#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

#specify order of items in legend
order = [
    labels.index("HAMMERv3"), 
    labels.index("HAMMERv2"), 
    labels.index("HAMMERv1"), 
    labels.index("IL + Random messages"), 
    labels.index("IL"), 
    labels.index("COIL")
    ]

#add legend to plot
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="lower right") 

plt.title('HAMMER (N='+str(N_AGENTS)+")") 
plt.xlabel("Number of Episodes")
plt.ylabel("Average Returns")
plt.xlim((0, CUT-SMOOTH)) 
plt.grid()
plt.show() 
    
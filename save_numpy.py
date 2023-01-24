import os, glob, numpy as np, matplotlib.pyplot as plt, pandas as pd 
from tensorboard.backend.event_processing import event_accumulator 
from collections import defaultdict


# LOG_DIR = "./logs"
# LOG_DIR = "./save_ablation_studies_final"
LOG_DIR = "./save_hammer_v1"

STORE_EVERYTHING_SIZE_GUIDANCE = {
    'compressedHistograms': 0, 
    'images': 0, 
    'audio': 0, 
    'scalars': 0, 
    'histograms': 0, 
} 

def get_values(filename, scalar="Episodic_Reward"): 
    ea = event_accumulator.EventAccumulator(filename, size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
    ea.Reload()
    # print(ea.Tags()) 
    ea_scalar = ea.Scalars(tag=scalar) 
    ea_scalar = pd.DataFrame(ea_scalar) 
    return ea_scalar 


logs = glob.glob(os.path.join(LOG_DIR, "*/**/event*"), recursive=True) 
for log in logs: 
    print(log) 
    vals = get_values(log, scalar="Avg_reward_for_each_agent__after_an_episode")['value'].to_numpy() 
    # vals = get_values(log, scalar="Episodic_Reward")['value'].to_numpy() 
    path = "/".join(log.split("/")[:-1]) 
    with open(path+'/arr.npy', 'wb') as f: 
        np.save(f, vals) 
    # break 

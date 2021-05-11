import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.cm as colormap, torch, json 

from itertools import count
from tensorboardX import SummaryWriter
from scipy.stats import entropy  
from pathlib import Path

from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_reference_v2

from hammer import PPO
from sum_guessing_game import GuessingSumEnv
from utils import read_config


def plot(obs, actions, messages, discrete_mes=False, nagents=2): 
    columns = ["index"] 
    for i in range(nagents): 
    
        columns.append("obs"+str(i+1)) 
        columns.append("action"+str(i+1)) 
        columns.append("message"+str(i+1)) 
    
    df = pd.DataFrame(columns=columns) 
    for k in range(len(obs)): 
        point=[0] 
        for i in range(nagents): 
            point.append(list(obs[k].values())[i][0]) 
            point.append(list(actions[k].values())[i][0]) 
            point.append(messages[k][i][0]) 
        df = df.append(pd.DataFrame([point], columns=columns)) 

    print(df) 
    if discrete_mes: 
        for x in range(nagents): 
            m = "message"+str(x+1) 
            df.loc[df[m] < 0.5, m] = 0 
            df.loc[df[m] >= 0.5, m] = 1 
    
    df["obs_sum"] = df["obs1"] + df["obs2"] + df["obs3"] 
    df["obs12_sum"] = df["obs1"] + df["obs2"] 
    df["obs13_sum"] = df["obs1"] + df["obs3"] 
    df["obs32_sum"] = df["obs3"] + df["obs2"] 

    # df["action1_err"] = abs(df["action1"] - df["obs_sum"]) 
    # df["action2_err"] = abs(df["action2"] - df["obs_sum"]) 
    # df["action3_err"] = abs(df["action3"] - df["obs_sum"]) 

    savedir = "plots/sumguessing/3agents_discrete_initial" 
    if not os.path.exists(savedir): os.makedirs(savedir)

    plots = [
        {
            "x": "obs1", 
            "y": "obs2", 
            "c": "message1" 
        }, 

        {
            "x": "obs1", 
            "y": "obs32_sum", 
            "c": "message1" 
        }, 

        {
            "x": "obs1", 
            "y": "obs_sum", 
            "c": "message1" 
        }, 

    ]

    for p in plots: 
        name = "--".join([p["x"],p["y"],p["c"]]) 
        df.plot.scatter(x=p["x"], y=p["y"], c=p["c"], colormap="viridis") 
        plt.savefig(os.path.join(savedir, name+".png")) 
        # plt.show() 

    # # Mean and Standard Deviation 
    # print(df["action1_err"].mean(), df["action1_err"].std()) # 0.7698002555758323, 0.5571670516844823 in 10000 eps 
    # print(df["action2_err"].mean(), df["action2_err"].std()) # 0.7838607136086739, 0.5664230119884279 in 10000 eps 
    

def run(args):
    
    SCALE = 10.0
    env = GuessingSumEnv(num_agents=args.nagents, scale=SCALE, discrete=0)

    env.reset()
    agents = env.agents

    obs_dim = 1
        
    action_dim = 1 

    config = read_config(args.config) 
    if not config:
        print("config required")
        return
    
    random_seed = args.randomseed 
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed) 
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    betas = (0.9, 0.999)

    HAMMER = PPO(
        agents=agents,
        single_state_dim=obs_dim, 
        single_action_dim=action_dim,
        meslen = args.meslen, 
        n_agents=len(agents), # required for discrete messages
        lr=config["lr"],
        betas=betas,
        gamma = config["gamma"],
        K_epochs=config["K_epochs"],
        eps_clip=config["eps_clip"],        
        actor_layer=config["actor_layer"],
        critic_layer=config["critic_layer"], 
        dru_toggle=args.dru_toggle, 
        is_discrete=config["is_discrete"], 
        sharedparams=0
    ) 
    HAMMER.load(args.load) 
    HAMMER.policy_old.action_var[0] = 1e-10 # For evaluation, reducing the variance in action (exploration) to close to zero 
    log_dir = Path('test/eval')
    for i in count(0):
        temp = log_dir/('run{}'.format(i)) 
        if temp.exists():
            pass
        else:
            writer = SummaryWriter(logdir=temp)
            log_dir = temp
            break
    
    if args.dru_toggle: 
        print("Using DRU") 
    else: 
        print("Not Using DRU")

    # logging variables
    ep_reward = 0
    global_timestep = 0

    env.reset() 

    obs_set = [] 
    action_set = [] 
    message_set = [] 
    diff_set = [] 
    ep_rew = [] 
    obs = env.reset() 

    i_episode = -1
    episode_rewards = 0
    for timestep in count(1):

        action_array = [] 

        actions, messages = HAMMER.policy_old.act(obs, HAMMER.memory, HAMMER.global_memory) 
        obs_set.append(obs)
        action_set.append(actions) 
        message_set.append(messages) 
        next_obs, rewards, is_terminals, infos = env.step(actions) 

        HAMMER.memory_record(rewards, is_terminals) 
        episode_rewards += np.mean(list(rewards.values())) 
        ep_rew.append(episode_rewards) 

        obs = next_obs

        # If episode had ended
        if all([is_terminals[agent] for agent in agents]):
            i_episode += 1
            writer.add_scalar('Avg reward for each agent, after an episode', episode_rewards, i_episode)
            obs = env.reset() 
            print('Episode {} \t  Avg reward for each agent, after an episode: {}'.format(i_episode, episode_rewards))
            episode_rewards = 0

        if i_episode == args.maxepisodes:
            break 


    # print(np.mean(ep_rew)) 

    plot(obs=obs_set, actions=action_set, messages=message_set, discrete_mes=args.dru_toggle, nagents=args.nagents) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/sumguessing.yaml', help="config file name")
    parser.add_argument("--load", type=str, default="save/guesser--nagents3--dru0--meslen1--rs--99/model_checkpoints/checkpoint_ep_10000") 

    parser.add_argument("--nagents", type=int, default=3)
    parser.add_argument("--maxepisodes", type=int, default=10000) 
    parser.add_argument("--dru_toggle", type=int, default=0) 
    parser.add_argument("--meslen", type=int, default=1, help="message length")
    parser.add_argument("--randomseed", type=int, default=10)

    args = parser.parse_args() 
    run(args=args)

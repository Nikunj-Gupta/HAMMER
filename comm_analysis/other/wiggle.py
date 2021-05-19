import argparse, os, numpy as np, torch, matplotlib.pyplot as plt 

from time import sleep
from pettingzoo.mpe import simple_spread_v2 

from hammer import PPO 
from utils import read_config 

def preprocess_one_obs(obs, which=1, limit=10): 
    agent = "agent_" + str(which) 
    obs[agent][limit:] = [0.]*(len(obs["agent_0"])-(limit)) 
    return obs 

def preprocess_obs(obs, limit): 
    for i in obs: 
        obs[i] = obs[i][:limit] 
    return obs 

def run(args): 
    env = simple_spread_v2.parallel_env(N=args.nagents, local_ratio=0.5, max_cycles=args.maxcycles) 
    env.reset()
    agents = [agent for agent in env.agents] 
    if args.heterogeneity: 
        obs_dim = len(preprocess_one_obs(env.reset(), limit=args.limit)["agent_0"]) 
    elif args.partialobs:
        obs_dim = len(preprocess_obs(env.reset(), limit=args.limit)["agent_0"]) 
    else:
        obs_dim = env.observation_spaces[env.agents[0]].shape[0]

    action_dim = env.action_spaces[env.agents[0]].n 
    agent_action_space = env.action_spaces[env.agents[0]] 
    
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
    if args.eval: 
        HAMMER.load(os.path.realpath(args.eval_path))

    if args.heterogeneity: 
        print('ERROR')
        obs = preprocess_one_obs(env.reset(), limit=args.limit) 
    elif args.partialobs: 
        print('ERROR')
        obs = preprocess_obs(env.reset(), limit=args.limit)
    else:  
        obs = env.reset() 

    W1 = 7 
    W2 = 8 

    # # For checking 
    # X = np.load(os.path.join("saves/save/eval--env_cn--n_3--dru_0--meslen_1--sharedparams_1--randomseed_9/dataset/hammer_states.npy"), mmap_mode="r") 
    # print(min(X[:,W1])) 
    # print(max(X[:,W1])) 
    # print(min(X[:,W2])) 
    # print(max(X[:,W2])) 
    
    # # exit()

    where = "plots/cn"+"/wiggle_plots" 
    if not os.path.exists(where): os.makedirs(where) 
    
    for agent in agents: 
        for i in range(0, 18, 2): 
            for message_index in range(3): 
                W1 = i 
                W2 = i+1 
                x = [] 
                y = [] 
                z = [] 

                for _ in range(10000):
                    obs[agent][W1] = np.random.uniform(-2, 2)
                    obs[agent][W2] = np.random.uniform(-2, 2)

                    actions, messages = HAMMER.policy_old.act(obs, HAMMER.memory, HAMMER.global_memory)  
                    
                    x.append(HAMMER.global_memory.states[0][0, (18*int(agent[-1]))+W1])
                    y.append(HAMMER.global_memory.states[0][0, (18*int(agent[-1]))+W2])
                    # message_index = 0 
                    z.append(HAMMER.global_memory.messages[0][message_index][0])
                    HAMMER.global_memory.clear_memory()

                sc = plt.scatter(x,y, c=z, cmap='RdYlBu') 
                plt.colorbar(sc) 
                name = "--".join([
                    agent, 
                    "W1_"+str(W1), 
                    "W2_"+str(W2), 
                    "message_index_"+str(message_index) 
                ])
                plt.savefig(os.path.join(where, name+".png")) 
                # plt.show() 
                plt.close() 
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/cn.yaml', help="config file name")

    parser.add_argument("--expname", type=str, default=None)
    parser.add_argument("--envname", type=str, default='cn')
    parser.add_argument("--nagents", type=int, default=3) 
    parser.add_argument("--eval", type=int, default=1) 
    parser.add_argument("--eval_path", type=str, default="saves/save/env_cn--n_3--dru_0--meslen_1--sharedparams_1--randomseed_9/model_checkpoints/checkpoint_ep_500000") 

    parser.add_argument("--sharedparams", type=int, default=1) 

    parser.add_argument("--maxepisodes", type=int, default=500_000) 
    parser.add_argument("--maxcycles", type=int, default=25) 
    parser.add_argument("--partialobs", type=int, default=0) 
    parser.add_argument("--heterogeneity", type=int, default=0) 
    parser.add_argument("--limit", type=int, default=10) # 10 for cn

    parser.add_argument("--dru_toggle", type=int, default=1) # 0 for HAMMERv2 and 1 for HAMMERv3 

    parser.add_argument("--meslen", type=int, default=1, help="message length")
    parser.add_argument("--randomseed", type=int, default=9)

    parser.add_argument("--saveinterval", type=int, default=10_000) 
    parser.add_argument("--logdir", type=str, default="logs/", help="log directory path")
    parser.add_argument("--savedir", type=str, default="model_checkpoints/", help="save directory path")
    

    args = parser.parse_args() 
    print(args.savedir)
    run(args=args) 

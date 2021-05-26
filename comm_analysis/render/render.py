import argparse, os, numpy as np, torch 
from numpy.lib.npyio import save
from numpy.lib.function_base import hamming 

from itertools import count
from tensorboardX import SummaryWriter
from pathlib import Path 
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
    if args.partialobs: print("Using Partial Observations") 
    
    if args.heterogeneity: print("Using Heterogeneous Local Agents") 

    env = simple_spread_v2.parallel_env(N=args.nagents, local_ratio=0.5, max_cycles=args.maxcycles) 
    env.reset() 
    agents = [agent for agent in env.agents] 
    if args.heterogeneity: obs_dim = len(preprocess_one_obs(env.reset(), limit=args.limit)["agent_0"]) 
    elif args.partialobs: obs_dim = len(preprocess_obs(env.reset(), limit=args.limit)["agent_0"]) 
    else: obs_dim = env.observation_spaces[env.agents[0]].shape[0]
    action_dim = env.action_spaces[env.agents[0]].n 

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
    HAMMER.load(args.eval_path) 

    if args.dru_toggle: print("Using DRU") 
    else: print("Not Using DRU")

    if args.heterogeneity: obs = preprocess_one_obs(env.reset(), limit=args.limit) 
    elif args.partialobs: obs = preprocess_obs(env.reset(), limit=args.limit)
    else: obs = env.reset() 

    # logging variables
    i_episode = 1 
    frames=[] 

    for timestep in count(0): 
        if timestep % args.maxcycles == 0: 
            if args.test_hammer == 1: log_dir = Path('./analysis-by-rendering/no_message') 
            else: log_dir = Path('./analysis-by-rendering/message')
            for i in count(1):
                temp = log_dir/('run{}'.format(i)) 
                if temp.exists(): pass
                else:
                    writer = SummaryWriter(logdir=temp)
                    log_dir = temp
                    break 

        frames.append(env.render(mode='rgb_array')) 
        
        actions, messages, _ = HAMMER.policy_old.act(obs, HAMMER.memory, HAMMER.global_memory, eval_zeros=args.test_hammer)  
        next_obs, rewards, is_terminals, infos = env.step(actions) 

        writer.add_scalar('reward', np.mean(np.array(list(rewards.values()))), timestep) 

        if args.partialobs: next_obs = preprocess_obs(next_obs, limit=args.limit) 
        elif args.heterogeneity: next_obs = preprocess_one_obs(next_obs, limit=args.limit) 
        obs = next_obs

        # recording all episodic messages of each agent 
        for i, agent in enumerate(agents): writer.add_scalar('message_' + str(agent), messages[i][0], timestep) 

        # If episode had ended
        if all([is_terminals[agent] for agent in agents]): 
            np.save(os.path.join(log_dir, "rgb_array.npy"), frames) 
            frames=[] 
            print('End of Episode {}'.format(i_episode)) 
            if args.heterogeneity: obs = preprocess_one_obs(env.reset(), limit=args.limit) 
            elif args.partialobs: obs = preprocess_obs(env.reset(), limit=args.limit)
            else: obs = env.reset() 

            i_episode += 1 

        if i_episode == args.maxepisodes+1: break
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser() 
    parser.add_argument("--config", type=str, default='configs/cn.yaml', help="config file name")

    parser.add_argument("--expname", type=str, default=None)
    parser.add_argument("--envname", type=str, default='cn')
    parser.add_argument("--nagents", type=int, default=3) 
    parser.add_argument("--eval", type=int, default=0) 
    parser.add_argument("--eval_path", type=str, default="") 

    parser.add_argument("--sharedparams", type=int, default=1) 

    parser.add_argument("--maxepisodes", type=int, default=100) 
    parser.add_argument("--maxcycles", type=int, default=25) 
    parser.add_argument("--partialobs", type=int, default=0) 
    parser.add_argument("--heterogeneity", type=int, default=0) 
    parser.add_argument("--limit", type=int, default=10) # 10 for cn

    parser.add_argument("--dru_toggle", type=int, default=1) # 0 for HAMMERv2 and 1 for HAMMERv3 

    parser.add_argument("--meslen", type=int, default=1, help="message length") 
    parser.add_argument("--test_hammer", type=int, default=0) 
    parser.add_argument("--randomseed", type=int, default=9)     

    args = parser.parse_args() 
    run(args=args) 

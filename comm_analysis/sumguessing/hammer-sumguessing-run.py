import argparse, os, numpy as np, torch, json 

from itertools import count
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_reference_v2
from tensorboardX import SummaryWriter
from pathlib import Path

from hammer import PPO
from utils import read_config
from sum_guessing_game import GuessingSumEnv

def run(args):
    
    SCALE = args.scale 
    env = GuessingSumEnv(num_agents=args.nagents, scale=SCALE, discrete=False)

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


    expname = "--".join([
        args.envname, 
        "nagents"+str(args.nagents), 
        "dru"+str(args.dru_toggle), 
        "meslen"+str(args.meslen),
        # "update_timestep", str(config["local"]["update_timestep"]),  
        # "lr", str(config["local"]["lr"]),  
        # "K_epochs", str(config["local"]["K_epochs"]), 
        "rs", str(args.randomseed), 
    ])
    
    writer = SummaryWriter(logdir=os.path.join("./save/", expname, args.logdir)) 
    # log_dir = Path(os.path.join(args.logdir, expname))
    # for i in count(0):
    #     temp = log_dir/('run{}'.format(i)) 
    #     if temp.exists():
    #         pass
    #     else:
    #         writer = SummaryWriter(logdir=temp)
    #         log_dir = temp
    #         break

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

    if args.dru_toggle: 
        print("Using DRU") 
    else: 
        print("Not Using DRU")

    # logging variables
    ep_reward = 0
    global_timestep = 0

    obs = env.reset() 

    i_episode = -1
    episode_rewards = 0
    ep_rew = []
    for timestep in count(1):

        action_array = [] 
        actions, messages = HAMMER.policy_old.act(obs, HAMMER.memory, HAMMER.global_memory) 
        next_obs, rewards, is_terminals, infos = env.step(actions) 

        HAMMER.memory_record(rewards, is_terminals) 
        episode_rewards += np.mean(list(rewards.values())) 
        ep_rew.append(episode_rewards) 
        # update if its time
        if timestep % config["update_timestep"] == 0: 
            HAMMER.update()
            [mem.clear_memory() for mem in HAMMER.memory]
            HAMMER.global_memory.clear_memory()

        obs = next_obs

        # If episode had ended
        if all([is_terminals[agent] for agent in agents]):
            i_episode += 1
            writer.add_scalar('Avg reward for each agent, after an episode', episode_rewards, i_episode)
            obs = env.reset() 
            print('Episode {} \t  Avg reward for each agent, after an episode: {}'.format(i_episode, episode_rewards))
            episode_rewards = 0

        # save every 50 episodes
        if i_episode % args.saveinterval == 0:
            save_dir = os.path.join("./save/", expname, args.savedir, "checkpoint_ep_"+str(i_episode)) 
        
            if not os.path.exists(save_dir): 
                os.makedirs(os.path.join(save_dir))  
            HAMMER.save(save_dir) 

        if i_episode == args.maxepisodes:
            break 
    print(np.mean(ep_rew))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/sumguessing.yaml', help="config file name")

    parser.add_argument("--expname", type=str, default=None)
    parser.add_argument("--envname", type=str, default='guesser')
    parser.add_argument("--nagents", type=int, default=3)
    parser.add_argument("--scale", type=float, default=10.0) 

    parser.add_argument("--maxepisodes", type=int, default=50_000) 

    parser.add_argument("--dru_toggle", type=int, default=0) 
    parser.add_argument("--sharedparams", type=int, default=0) 

    parser.add_argument("--meslen", type=int, default=1, help="message length")
    parser.add_argument("--randomseed", type=int, default=10)

    parser.add_argument("--saveinterval", type=int, default=10_000) 
    parser.add_argument("--logdir", type=str, default="logs/", help="log directory path")
    parser.add_argument("--savedir", type=str, default="model_checkpoints", help="save directory path")
    

    args = parser.parse_args() 
    print(args.savedir)
    run(args=args)

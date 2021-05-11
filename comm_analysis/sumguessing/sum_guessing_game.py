import numpy as np


class GuessingSumEnv:
    def __init__(self, num_agents=5, discrete = False, scale=10.0):
        self.num_agents = num_agents
        self.discrete = discrete
        self.observations = []
        self.scale = scale
        self.sum_scale = self.num_agents * self.scale
        self.agents = ["Agent{}".format(i) for i in range(self.num_agents)] 

    def step(self, actions):
        actions = list(actions.values()) 

        observations = None
        rewards = -np.abs(actions - self.sum) # [-Inf ; 0]

        # normalized_rewards = (np.maximum(rewards, -self.sum_scale) + self.sum_scale) / self.sum_scale # [0 ; 1]
        normalized_rewards = rewards

        done = {}
        info = None

        rewards = {}
        for i, agent in enumerate(self.agents):
            rewards[agent] = normalized_rewards[i]
            done[agent] = True

        return observations, rewards, done, info

    def reset(self):
        if self.discrete:
            observations = np.random.randint(low=0, high= self.scale, size=(self.num_agents, 1))
        else: 
            # observations = np.clip(np.random.normal(size=(self.num_agents, 1)), -self.scale, self.scale) 
            observations = np.random.uniform(-self.scale, self.scale,size=(self.num_agents, 1)) 
        self.observations = observations.reshape(-1) 
        self.sum = sum(self.observations) # np.prod(self.observations) 
        obs = {}
        for i, agent in enumerate(self.agents):
            obs[agent] = np.array(observations[i])
        return obs

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed=None):
        np.random.seed(seed)
        return


if __name__ == '__main__':
    env = GuessingSumEnv(num_agents=2)
    env.seed(0)

    print('obs:', env.reset())
    actions = {}
    for agent in env.agents:
        actions[agent] = np.random.uniform(-env.scale, env.scale, size=1)
    print('actions:', actions)
    print('Step Returns:', env.step(actions))
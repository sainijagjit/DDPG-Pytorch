import os
import gym
import json
import torch
import numpy as np
from tqdm import tqdm
from src.utils import *
from src.memory import *
from src.agents import *


class Trainer:

    def __init__(self, config_file, enable_logging=False):
        self.enable_logging = enable_logging
        self.config = Trainer.parse_config(config_file)
        self.env = gym.make(self.config['env_name'])
        self.apply_seed()
        self.state_dimension = self.env.observation_space.shape[0]
        self.action_dimension = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.agent = DDPGAgent(
            state_dim=self.state_dimension, action_dim=self.action_dimension,
            max_action=self.max_action, device=self.device,
            discount=self.config['discount'], tau=self.config['tau']
        )
        self.save_file_name = f"DDPG_{self.config['env_name']}_{self.config['seed']}"
        self.memory = ReplayBuffer()
        # if self.enable_logging:
        #     from torch.utils.tensorboard import SummaryWriter
        #     self.writer = SummaryWriter('./logs/' + self.config['env_name'] + '/')
        try:
            os.mkdir('./models')
        except Exception as e:
            pass

    @staticmethod
    def parse_config(json_file):
        with open(json_file, 'r') as f:
            configs = json.load(f)
        return configs

    def apply_seed(self):
        if hasattr(self.env, 'seed'):
            self.env.seed(self.config['seed'])
        else:
            self.env.action_space.seed(self.config['seed'])
            self.env.observation_space.seed(self.config['seed'])

        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])

    def train(self):
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        evaluations = []
        episode_rewards = []
        for ts in tqdm(range(1, int(self.config['time_steps']) + 1)):
            episode_timesteps += 1
            if ts < self.config['start_time_step']:
                action = self.env.action_space.sample()
            else:
                action = (
                        self.agent.select_action(np.array(state)) + np.random.normal(
                    0, self.max_action * self.config['expl_noise'],
                    size=self.action_dimension
                )
                ).clip(
                    -self.max_action,
                    self.max_action
                )
            next_state, reward, done, trunc, info = self.env.step(action)
            self.memory.push(
                state, action, next_state, reward,
                float(done) if episode_timesteps < self.env._max_episode_steps else 0)
            state = next_state
            episode_reward += reward
            if ts >= self.config['start_time_step']:
                self.agent.train(self.memory, self.config['batch_size'])
            if done:
                # if self.enable_logging:
                #     self.writer.add_scalar('Episode Reward', episode_reward, ts)
                episode_rewards.append(episode_reward)
                state = self.env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
        if ts % 1000 == 0:
            evaluations.append(evaluate_policy(self.agent, self.config['env_name'], self.config['seed']))
            self.agent.save_checkpoint(f"./models/{self.save_file_name}")
        return episode_rewards, evaluations
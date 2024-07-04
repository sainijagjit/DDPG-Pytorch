import os
import wandb
import gym
import json
import torch
import numpy as np
from tqdm import tqdm
from src.utils import *
from src.memory import *
from src.agents import *

os.environ['WANDB_API_KEY'] = ''
class Trainer:

    def __init__(self, config_file, enable_logging=True):
        self.enable_logging = enable_logging
        self.config = Trainer.parse_config(config_file)
        self.env = gym.make(self.config['env_name'])
        self.env = apply_seed(self.env,self.config['seed'])
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
        if self.enable_logging:
            wandb.init(project="ddpg", config=self.config)
        try:
            os.mkdir('./pretrained_models')
        except Exception as e:
            pass

    @staticmethod
    def parse_config(json_file):
        with open(json_file, 'r') as f:
            configs = json.load(f)
        return configs

    def train(self):
        state, info = self.env.reset()
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
                state, action,reward, next_state,
                float(done) if episode_timesteps < self.env._max_episode_steps else 0)
            state = next_state
            episode_reward += reward
            if ts >= self.config['start_time_step']:
                self.agent.train(self.memory, self.config['batch_size'])
            if done:
                if self.enable_logging:
                    wandb.log({'Episode Reward': episode_reward, 'Timesteps': ts})
                episode_rewards.append(episode_reward)
                state, info = self.env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                if episode_num % 100 == 0 and episode_num > 0:
                    evaluations.append(evaluate_policy(self.agent, self.config['env_name'], self.config['seed'],enable_logging=self.enable_logging,wandb=wandb))
                    self.agent.save_checkpoint(f"./pretrained_models/{self.save_file_name}")
                episode_num += 1
        wandb.finish()
        return episode_rewards, evaluations
    
    def evaluate(self):
        self.agent.load_checkpoint(f"./pretrained_models/DDPG_{self.config['env_name']}_{self.config['seed']}")
        evaluate_policy(self.agent, self.config['env_name'], self.config['seed'],render=True)
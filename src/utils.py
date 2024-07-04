import gym
import torch
import numpy as np

def apply_seed(env,seed):
    if hasattr(env, 'seed'):
        env.seed(seed)
    else:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return env


def evaluate_policy(policy, env_name, seed, wandb=False,enable_logging=True, eval_episodes=10, render=False):
    if render:
        eval_env = gym.make(env_name, render_mode='human')
    else:
        eval_env = gym.make(env_name)
    eval_env = apply_seed(eval_env,seed)
    avg_reward = 0
    for _ in range(eval_episodes):
        episode_reward=0
        state, info = eval_env.reset()
        done = False
        trunc = False
        while not done and not trunc:
            action = policy.select_action(np.array(state))
            if render:
                eval_env.render()
            state, reward, done, trunc, info = eval_env.step(action)
            avg_reward += reward
            episode_reward += reward
        print(f'Episode reward {_+1} ==> {episode_reward}')
    avg_reward /= eval_episodes
    if enable_logging:
        wandb.log({'Test Episode Reward': avg_reward})
    else:
        print('avg_reward',avg_reward)
    return avg_reward


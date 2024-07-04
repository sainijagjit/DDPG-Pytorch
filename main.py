from train import Trainer
from matplotlib import pyplot as plt


env_name='BipedalWalker-v3'
trainer = Trainer(config_file=f'./configs/{env_name}.json',enable_logging=True)
episode_rewards, evaluations = trainer.train()

trainer.agent.save_checkpoint(f'./pretrained_models/{env_name}')

plt.figure(figsize=(16, 10))
plt.plot(episode_rewards)
from train import Trainer
from matplotlib import pyplot as plt

trainer = Trainer(config_file='./configs/BipedalWalker-v3.json')
episode_rewards, evaluations = trainer.train()

trainer.agent.save_checkpoint('./pretrained_models/BipedalWalker-v3')

plt.figure(figsize=(16, 10))
plt.plot(episode_rewards)
from train import Trainer
from matplotlib import pyplot as plt


env_name='BipedalWalker-v3'
trainer = Trainer(config_file=f'./configs/{env_name}.json',enable_logging=False)
trainer.evaluate()


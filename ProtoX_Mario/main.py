import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # this might cause crash/error
from ProtoX_Mario.models import *

from ProtoX_Mario.utils import *
from ProtoX_Mario.pytorchtools import EarlyStopping

# from quadruplet import *

import torch.optim as optim
from tqdm import tqdm
import pandas as pd

new_data = False


def data_generation_or_loading():
    # get training data loaders
    if new_data:
        expert_observations, color_observations, \
        expert_actions, episode_schedule = gen_color_data(output_path=None,num_interactions=int(1e4),bad=True)  # =int(30000/322))
        np.savez_compressed(
            'mario_HOLDOUT.npz',
            expert_actions=expert_actions,  # np.array(acts),
            color_observations=color_observations,
            expert_observations=expert_observations,  # np.array(states),
            episode_schedule=episode_schedule  # np.array(episode_schedule)
        )
    else:
        arrs = np.load('marioCROP.npz')
        expert_observations = arrs['expert_observations']
        color_observations = arrs['color_observations']
        expert_actions = arrs['expert_actions']
        episode_schedule = arrs['episode_schedule']

        expert_dataset = ExpertDataset(expert_observations, expert_actions)
        episode_labels = torch.FloatTensor(episode_schedule[:, 0]).to(device)
        step_labels = torch.FloatTensor(episode_schedule[:, 1]).to(device)
        DatasetWithInDices = dataset_with_indices(ExpertDataset)
        dset = DatasetWithInDices(expert_observations, expert_actions)


if __name__ == '__main__':
    print("ipynb to py mario")
    if torch.cuda.is_available:
        device = "cuda:0"
        print('Using GPU')
    else:
        device = "cpu"
        print('using CPU')

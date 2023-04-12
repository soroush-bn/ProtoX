import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # this might cause crash/error
from ProtoX_Mario.models import *

from ProtoX_Mario.utils import *
from ProtoX_Mario.pytorchtools import EarlyStopping

# from quadruplet import *
import torch as th
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

new_data = True


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
    return expert_dataset,episode_labels,step_labels,dset


def get_evaluation_data():
    # get evaluation data
    holdout_arrs = np.load('marioCROP_HOLDOUT.npz')
    holdout_expert_observations = holdout_arrs['expert_observations']
    holdout_color_observations = holdout_arrs['color_observations']
    holdout_expert_actions = holdout_arrs['expert_actions']
    holdout_episode_schedule = holdout_arrs['episode_schedule']

    holdout_expert_dataset = ExpertDataset(holdout_expert_observations, holdout_expert_actions)

    eval_loader = th.utils.data.DataLoader(
        dataset=holdout_expert_dataset, batch_size=128, shuffle=False
    )

if __name__ == '__main__':
    print("ipynb to py mario \n ---------------------------")
    if torch.cuda.is_available:
        device = "cuda:0"
        print('Using GPU')
    else:
        device = "cpu"
        print('using CPU')

    expert_dataset, episode_labels, step_labels, dset = data_generation_or_loading()
    print("finished")

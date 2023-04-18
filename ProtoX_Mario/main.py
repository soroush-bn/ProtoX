import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # this might cause crash/error
from ProtoX_Mario.models import *
from matplotlib import pyplot as plt
from ProtoX_Mario.utils import *
from ProtoX_Mario.pytorchtools import EarlyStopping

# from quadruplet import *
import torch as th
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from .quadrupletloss import *
new_data = True

# expert_dataset = None
# episode_labels = None
# step_labels = None
# DatasetWithInDices = None
# dset = None


def data_generation_or_loading():
    # get training data loaders
    if new_data:
        expert_observations, color_observations, \
        expert_actions, episode_schedule = gen_color_data(output_path=None, num_interactions=int(1e4),
                                                          bad=True)  # =int(30000/322))
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
    return expert_dataset, episode_labels, step_labels, dset


holdout_arrs = None
holdout_expert_observations = None
holdout_color_observations = None
holdout_expert_actions = None
holdout_episode_schedule = None
holdout_expert_dataset = None
eval_loader = None


def get_evaluation_data():
    # get evaluation data
    holdout_arrs = np.load('mario_HOLDOUT.npz')
    holdout_expert_observations = holdout_arrs['expert_observations']
    holdout_color_observations = holdout_arrs['color_observations']
    holdout_expert_actions = holdout_arrs['expert_actions']
    holdout_episode_schedule = holdout_arrs['episode_schedule']

    holdout_expert_dataset = ExpertDataset(holdout_expert_observations, holdout_expert_actions)

    eval_loader = th.utils.data.DataLoader(
        dataset=holdout_expert_dataset, batch_size=128, shuffle=False
    )


def encoder_pretraining():
    # dataset for encoder pretraining
    # RUN THIS FOR PRE-TRAINING
    train_size = int(0.8 * len(dset))
    test_size = int(0.2 * len(dset))

    train_expert_dataset, test_expert_dataset = random_split(
        dset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    kwargs = {"num_workers": 8, "pin_memory": False}
    train_loader = th.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=64, shuffle=True, **kwargs
    )
    test_loader = th.utils.data.DataLoader(
        dataset=test_expert_dataset, batch_size=64, shuffle=True, **kwargs,
    )
    return  train_loader, test_loader


def plot_ae_outputs(test_loader,train_loader,encoder, decoder, n=10):
    plt.figure(figsize=(16, 4.5))
    try:
        imgs, targets = next(iter(test_loader))
    except Exception as e:
        imgs, targets, _ = next(iter(test_loader))

    targets = targets.numpy()

    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = imgs[i].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img = decoder(encoder(img))
        for k in range(4):
            plt.imshow(img.cpu().squeeze().numpy()[k], cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, i + 1 + n)
        for k in range(4):
            plt.imshow(rec_img.cpu().squeeze().numpy()[k], cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed images')
    plt.show()


### Training function
def train_siamese_epoch(vae, device, dataloader, optimizer, criterion, beta=1., lbda=.1):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    vae_loss_tot = 0.0
    siam_loss_tot = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for batch, (x, actions, idx) in enumerate(dataloader, 1):
        # Move tensor to the proper device
        x = x.to(device)
        actions = actions.to(device)
        idx = idx.to(device)

        # forward pass
        x_hat, z = vae(x)

        # Evaluate VAE loss
        vae_loss = ((x - x_hat) ** 2).sum() + beta * vae.encoder.kl

        # Evaluate triplet/quadruplet loss
        siam_loss = criterion(embeddings=z, labels=actions, idx=idx)

        loss = vae_loss + lbda * siam_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.item()))

        train_loss += loss.item()
        vae_loss_tot += vae_loss.item()
        siam_loss_tot += siam_loss.item()

    print('Avg VAE/SIAM loss: ', vae_loss_tot / len(dataloader.dataset), siam_loss_tot / len(dataloader.dataset))

    return train_loss / len(dataloader.dataset)

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

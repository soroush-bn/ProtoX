# https://raw.githubusercontent.com/lyakaap/NetVLAD-pytorch/master/hard_triplet_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardQuadrupletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)

    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """

    def __init__(self, margin1=0.1, margin2=.05, hardest=False, squared=False, epsilon=10, tau=11):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardQuadrupletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.hardest = hardest
        self.squared = squared
        self.epsilon = epsilon
        self.tau = tau

    def forward(self, embeddings, labels, idx):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        pairwise_dist = _pairwise_distance(embeddings, squared=self.squared)

        if self.hardest:
            # Get the hardest positive pairs
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1, keepdim=True)

            # Get the hardest negative1 pairs
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
            max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                    1.0 - mask_anchor_negative)
            hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

            # Get hardest negative 2 pairs

            # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
            quad_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin1)
            quad_loss += F.relu(hardest_positive_dist - hardest_negative2_dist + self.margin2)
            quad_loss = torch.mean(quad_loss)
        else:
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)
            anc_neg2_dist = pairwise_dist.unsqueeze(dim=0)

            # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
            # triplet_loss[i, j, k] will contain the triplet loss of anc=i, pos=j, neg=k
            # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
            # and the 2nd (batch_size, 1, batch_size)
            loss = F.relu(anc_pos_dist - anc_neg_dist + self.margin1)
            loss += F.relu(anc_pos_dist - anc_neg2_dist + self.margin2)

            # print('\ninit loss stats', torch.min(loss).item(), torch.max(loss).item())

            mask = _get_quadruplet_mask(labels, idx).float()
            quadruplet_loss = loss * mask

            # print('masked loss stats', torch.min(triplet_loss).item(), torch.max(triplet_loss).item())

            # Remove negative losses (i.e. the easy triplets)
            # quadruplet_loss = F.relu(quadruplet_loss)

            # Count number of hard triplets (where triplet_loss > 0)
            hard_quadruplets = torch.gt(quadruplet_loss, 1e-16).float()
            num_hard_quadruplets = torch.sum(hard_quadruplets)

            quadruplet_loss = torch.sum(quadruplet_loss) / (num_hard_quadruplets + 1e-16)

        return quadruplet_loss


def _pairwise_distance(x, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    cor_mat = torch.matmul(x, x.t())
    norm_mat = cor_mat.diag()
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    distances = F.relu(distances)

    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1

    # Check if labels[i] == labels[j]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

    mask = indices_not_equal * labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels):
    # Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    # Check if labels[i] != labels[k]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ 1

    return mask


def _get_quadruplet_mask(episode_labels,
step_labels,
        labels,
        idx
):
    B = labels.size(0)

    # Make sure that i != j != k != l
    indices_equal = torch.eye(B, dtype=torch.bool).to(torch.device)  # [B, B]
    indices_not_equal = ~indices_equal  # [B, B]
    i_not_equal_j = indices_not_equal.view(B, B, 1, 1)  # [B, B, 1, 1]
    j_not_equal_k = indices_not_equal.view(1, B, B, 1)  # [B, 1, 1, B]
    k_not_equal_l = indices_not_equal.view(1, 1, B, B)  # [1, 1, B, B]
    distinct_indices = i_not_equal_j & j_not_equal_k & k_not_equal_l  # [B, B, B, B]

    # Make sure that labels[i] == labels[j]
    #            and labels[j] != labels[k]
    #            and labels[k] != labels[l]
    labels_equal = labels.view(1, B) == labels.view(B, 1)  # [B, B]
    i_equal_j = labels_equal.view(B, B, 1, 1)  # [B, B, 1, 1]
    j_equal_k = labels_equal.view(1, B, B, 1)  # [1, B, B, 1]
    l_equal_i = labels_equal.view(B, 1, 1, B)  # [1, 1, B, B]
    label_match = i_equal_j & ~j_equal_k  # & ~l_equal_i

    eps = 15  # self.epsilon
    tau = 15  # self.tau

    ep_labels = episode_labels[idx]
    lst = step_labels[idx]

    ep_labels_equal = ep_labels.view(1, B) == ep_labels.view(B, 1)  # [B, B]
    i_equal_j_ep = labels_equal.view(B, B, 1, 1)  # [B, B, 1, 1]
    j_equal_k_ep = labels_equal.view(1, B, B, 1)  # [1, B, B, 1]
    k_equal_l_ep = labels_equal.view(1, 1, B, B)  # [1, 1, B, B]
    episode_match = i_equal_j_ep & j_equal_k_ep & k_equal_l_ep

    # uncomment this to keep quadruplets in the same episode
    '''
    within_eps_over_tau = torch.logical_and(torch.abs(lst[:, None, None, None] - lst[None, :, None, None]) <= eps,
                                            torch.abs(lst[:,None,None,None]-lst[None,None,:,None]) <= eps,
                                            torch.abs(lst[:, None, None,None] - lst[None, None,None,:]) >= tau)
    '''
    within_eps_over_tau = (torch.abs(lst[:, None, None, None] - lst[None, :, None, None]) <= eps) & (
        torch.abs(lst[:, None, None, None] - lst[None, None, :, None] <= eps)) & (
                                      torch.abs(lst[:, None, None, None] - lst[None, None, None, :]) >= tau)
    # return within_eps_over_tau & episode_match & label_match & distinct_indices  # [B, B, B, B]
    return within_eps_over_tau & label_match & distinct_indices

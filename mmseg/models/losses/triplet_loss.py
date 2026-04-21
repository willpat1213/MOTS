# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS


@MODELS.register_module()
class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Args:
        margin (float, optional): Margin for triplet loss. Default to 0.3.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
    """

    def __init__(self, margin=0.3, loss_weight=1.0, mode='batch_all', constra_feat_dim=256, eps=1e-12, loss_name='loss_triplet'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_weight = loss_weight
        self.constra_feat_dim=constra_feat_dim,
        self.mode = mode
        self._loss_name = loss_name
        self.eps = eps # 保证 numerical stability



    def euclidean_distance_matrix(self, inputs):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape
                (batch_size, feat_dim).
        """

        batch_size = inputs.size(0)

        # Compute Euclidean distance
        dist = torch.pow(inputs, 2).sum(
            dim=1, keepdim=True).expand(batch_size, batch_size)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        return dist
    
    def get_triplet_mask(self, labels):
        # step 1 - keep different samples in triplet.
        # (batch_size, batch_size)
        indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
        indices_not_equal = torch.logical_not(indices_equal)
        # (batch_size, batch_size, 1)
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        # (batch_size, 1, batch_size)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        # (1, batch_size, batch_size)
        j_not_equal_k = indices_not_equal.unsqueeze(0)
        # (batch_size, batch_size, batch_size)
        distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

        # step 2 - keep anchor and positive have same label，and negative have different label.
        # (batch_size, batch_size)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        # (batch_size, batch_size, 1)
        i_equal_j = labels_equal.unsqueeze(2)
        # (batch_size, 1, batch_size)
        i_equal_k = labels_equal.unsqueeze(1)
        # (batch_size, batch_size, batch_size)
        valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

        # step 3 - keep both two pass.
        mask = torch.logical_and(distinct_indices, valid_indices)

        return mask

    def forward(self, inputs, data_samples, **kwargs):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape
                (batch_size, channels, height, width).
            labels (torch.LongTensor): ground truth labels with shape
                (batch_size, ).
        """
        last_feat = inputs[-1]
        b, c, h, w = last_feat.shape
        dim_in = c * h * w
        # last_feat = last_feat.view(b, -1)
        # mlp = nn.Sequential(
            # nn.Flatten(),
            # nn.Linear(dim_in, dim_in),
            # nn.ReLU(inplace=True),
            # nn.Linear(dim_in, self.constra_feat_dim)
        # )
        avg = nn.AdaptiveAvgPool2d((1, 1))
        fc = nn.Linear(c, self.constra_feat_dim[0]).to(last_feat.device)

        # (batch_size, feat_dims)
        last_feat = avg(last_feat)
        last_feat = last_feat.view(b, -1)
        last_feat = fc(last_feat)
        feat_in = F.normalize(last_feat, dim=1)

        # (batch_size, batch_size)
        distance_matrix = self.euclidean_distance_matrix(feat_in)

        labels = []
        self.data_samples = data_samples
        labels = [data.overlap_label for data in data_samples]
        labels = torch.tensor(labels, device=feat_in.device)
        
        loss_dict = dict()

        if self.mode == 'batch_all':
            # (batch_size, batch_size, 1)
            anchor_positive_dists = distance_matrix.unsqueeze(2)
            # (batch_size, 1, batch_size)
            anchor_negative_dists = distance_matrix.unsqueeze(1)
            # (batch_size, batch_size, batch_size)
            triplet_loss = anchor_positive_dists - anchor_negative_dists + self.margin
            
            mask = self.get_triplet_mask(labels)
            triplet_loss *= mask
            triplet_loss = F.relu(triplet_loss)
            num_positive_losses = (triplet_loss > self.eps).float().sum()
            triplet_loss = triplet_loss.sum() / (num_positive_losses + self.eps)

        elif self.mode == 'batch_hard':
            # get hard positive 和 negative
            hardest_positive_dist = distance_matrix.max(dim=1, keepdim=True)[0]
            hardest_negative_dist = distance_matrix.min(dim=1, keepdim=True)[0]
            triplet_loss = hardest_positive_dist - hardest_negative_dist + self.margin
            triplet_loss = F.relu(triplet_loss)
            triplet_loss = triplet_loss.mean()

        elif self.mode == 'batch_semihard':
            # get positive 和 semihard negative
            hardest_positive_dist = distance_matrix.max(dim=1, keepdim=True)[0]
            semihard_negative_dist = (distance_matrix + self.margin).min(dim=1, keepdim=True)[0]
            triplet_loss = hardest_positive_dist - semihard_negative_dist + self.margin
            triplet_loss = F.relu(triplet_loss)
            triplet_loss = triplet_loss.mean()

        loss_dict['loss_triplet'] = triplet_loss
        return loss_dict

        
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
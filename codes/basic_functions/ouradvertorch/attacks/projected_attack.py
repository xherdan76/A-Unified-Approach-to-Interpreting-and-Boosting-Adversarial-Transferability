from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy

import numpy as np
import torch
import torch.nn as nn

from ..utils import clamp, normalize_by_pnorm, rand_init_delta
from .interaction_loss import (InteractionLoss, get_features,
                               sample_for_interaction)


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def transition_invariant_conv(size=15):
    kernel = gkern(size, 3).astype(np.float32)
    padding = size // 2
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)

    conv = nn.Conv2d(
        in_channels=3,
        out_channels=3,
        kernel_size=size,
        stride=1,
        groups=3,
        padding=padding,
        bias=False)
    conv.weight.data = conv.weight.new_tensor(data=stack_kernel)

    return conv


class ProjectionAttacker(object):

    def __init__(self,
                 model,
                 epsilon,
                 num_steps,
                 step_size,
                 ord='inf',
                 image_width=224,
                 loss_fn=None,
                 targeted=False,
                 grid_scale=8,
                 sample_times=32,
                 sample_grid_num=32,
                 momentum=0.0,
                 ti_size=1,
                 lam=1,
                 m=0,
                 sigma=15,
                 rand_init=True):
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.image_width = image_width
        self.momentum = momentum
        self.targeted = targeted
        self.ti_size = ti_size
        self.lam = lam
        self.grid_scale = grid_scale
        self.sample_times = sample_times
        if self.ti_size > 1:
            self.ti_conv = transition_invariant_conv(ti_size)
        self.sample_grid_num = sample_grid_num
        self.m = m
        self.sigma = sigma
        self.ord = ord
        self.rand_init = rand_init
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

    def perturb(self, X, y):
        """
        :param X_nat: a Float Tensor
        :param y: a Long Tensor
        :return:
        """
        loss_record = {'loss1': [], 'loss2': [], 'loss': []}
        delta = torch.zeros_like(X)
        if self.rand_init and self.lam == 0:
            rand_init_delta(delta, X, self.ord, self.epsilon, 0.0, 1.0)
            delta.data = clamp(X + delta.data, min=0.0, max=1.0) - X

        delta.requires_grad_()

        grad = torch.zeros_like(X)
        deltas = torch.zeros_like(X).repeat(self.num_steps, 1, 1, 1)
        label = y.item()

        noise_distribution = torch.distributions.normal.Normal(
                    torch.tensor([0.0]),
                    torch.tensor([self.sigma]).float())

        for i in range(self.num_steps):
            if self.m >= 1:  # Variance-reduced attack; https://arxiv.org/abs/1802.09707
                noise_shape = list(X.shape)
                noise_shape[0] = self.m
                noise = noise_distribution.sample(noise_shape).squeeze() / 255
                noise = noise.to(X.device)
                outputs = self.model(X + delta + noise)

                loss1 = self.loss_fn(outputs, y.expand(self.m))
            else:
                loss1 = self.loss_fn(self.model(X + delta), y)

            if self.targeted:
                loss1 = -loss1

            if self.lam > 0:  # Interaction-reduced attack
                only_add_one_perturbation, leave_one_out_perturbation = \
                    sample_for_interaction(delta, self.sample_grid_num,
                                           self.grid_scale, self.image_width,
                                           self.sample_times)

                (outputs, leave_one_outputs, only_add_one_outputs,
                 zero_outputs) = get_features(self.model, X, delta,
                                              leave_one_out_perturbation,
                                              only_add_one_perturbation)

                outputs_c = copy.deepcopy(outputs.detach())
                outputs_c[:, label] = -np.inf
                other_max = outputs_c.max(1)[1].item()
                interaction_loss = InteractionLoss(
                    target=other_max, label=label)
                average_pairwise_interaction = interaction_loss(
                    outputs, leave_one_outputs, only_add_one_outputs,
                    zero_outputs)

                if self.lam == float('inf'):
                    loss2 = -average_pairwise_interaction
                    loss = loss2
                else:
                    loss2 = -self.lam * average_pairwise_interaction
                    loss = loss1 + loss2

                loss_record['loss1'].append(loss1.item())
                loss_record['loss2'].append(
                    loss2.item() if self.lam > 0 else 0)
                loss_record['loss'].append(loss.item())
            else:
                loss = loss1
            loss.backward()

            deltas[i, :, :, :] = delta.data

            cur_grad = delta.grad.data
            if self.ti_size > 1:  # TI Attack; https://arxiv.org/abs/1904.02884
                self.ti_conv.to(X.device)
                cur_grad = self.ti_conv(cur_grad)

            # MI Attack; https://arxiv.org/abs/1710.06081
            cur_grad = normalize_by_pnorm(cur_grad, p=1)
            grad = self.momentum * grad + cur_grad
            
            if self.ord == np.inf:
                delta.data += self.step_size * grad.sign()
                delta.data = clamp(delta.data, -self.epsilon, self.epsilon)
                delta.data = clamp(X.data + delta.data, 0.0, 1.0) - X.data
            elif self.ord == 2:
                delta.data += self.step_size * normalize_by_pnorm(grad, p=2)
                delta.data *= clamp(
                    (self.epsilon * normalize_by_pnorm(delta.data, p=2) /
                     delta.data),
                    max=1.)
                delta.data = clamp(X.data + delta.data, 0.0, 1.0) - X.data
            else:
                error = "Only ord = inf and ord = 2 have been implemented"
                raise NotImplementedError(error)

            delta.grad.data.zero_()
        rval = X.data + deltas
        return rval, loss_record

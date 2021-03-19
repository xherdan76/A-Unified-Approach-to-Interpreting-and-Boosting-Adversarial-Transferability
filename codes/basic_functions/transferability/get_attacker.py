import numpy as np
import torch.nn as nn
from codes.basic_functions.ouradvertorch.attacks.projected_attack import \
    ProjectionAttacker
from codes.utils import register_hook_for_densenet, register_hook_for_resnet


def get_attacker(
    attack_method,
    arch,
    predict,
    p,
    epsilon,
    num_steps,
    step_size,
    image_dim,
    image_size,
    grid_scale,
    sample_grid_num,
    sample_times,
    momentum=0.,
    gamma=1.,
    lam=0.,
    ti_size=1,
    m=0,
    sigma=15,
):

    if ('SGM' in attack_method or 'Hybrid' in attack_method) and gamma > 1:
        raise Exception('gamma of SGM method should be less than 1')
    if ('MI' in attack_method or 'Hybrid' in attack_method) and momentum == 0:
        raise Exception('momentum of MI method should be greater than 0')
    if ('VR' in attack_method or 'Hybrid' in attack_method) and m == 0:
        raise Exception('m of VR method should be greater than 0')
    if ('TI' in attack_method) and ti_size == 1:
        raise Exception('ti_size of  the TI method should be greater than 0')
    if ('IR' in attack_method or 'Hybrid' in attack_method) and lam == 0:
        raise Exception('lam of  the IR method should be greater than 0')

    if p == 'inf':
        p = np.inf
        epsilon = epsilon / 255.
        step_size = step_size / 255.
        num_steps = num_steps
    elif int(p) == 2:
        p = 2
        epsilon = epsilon / 255. * np.sqrt(image_dim)
        step_size = float(step_size)
        num_steps = num_steps
    else:
        raise NotImplementedError('p should be inf or 2')

    # set for SGM Attack
    if gamma < 1.0 and ('SGM' in attack_method or 'Hybrid' in attack_method):
        if arch in [
                'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        ]:
            register_hook_for_resnet(predict, arch=arch, gamma=gamma)
        elif arch in ['densenet121', 'densenet169', 'densenet201']:
            register_hook_for_densenet(predict, arch=arch, gamma=gamma)
        else:
            raise ValueError(
                'Current code only supports resnet/densenet. '
                'You can extend this code to other architectures.')

    adversary = ProjectionAttacker(
        model=predict,
        epsilon=epsilon,
        num_steps=num_steps,
        step_size=step_size,
        image_width=image_size,
        momentum=momentum,
        ti_size=ti_size,
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        targeted=False,
        lam=lam,
        grid_scale=grid_scale,
        sample_times=sample_times,
        sample_grid_num=sample_grid_num,
        m=m,
        sigma=sigma,
        ord=p)
    return adversary

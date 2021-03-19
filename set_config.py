import os

from codes.utils import seed_torch


def set_config(args):
    setting = os.path.join(
        f'source_{args.arch}', f'L_{args.p}_eps_{args.epsilon}',
        '_'.join([args.attack_method, f'lam_{args.lam}_seed_{args.seed}']))

    args.adv_image_root = os.path.join(args.adv_image_root, setting)
    args.loss_root = os.path.join(args.loss_root, setting)

    seed = args.seed
    for c in setting:
        seed += ord(c)
    seed_torch(seed)

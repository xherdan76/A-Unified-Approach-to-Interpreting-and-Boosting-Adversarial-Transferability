import argparse

from codes.basic_functions.transferability import (interaction_reduced_attack,
                                                   leave_one_out)
from set_config import set_config

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--p", type=str, help="inf; 2", default='inf')
parser.add_argument("--epsilon", type=int, default=16)
parser.add_argument("--step_size", type=float, default=2.)
parser.add_argument("--num_steps", type=int, default=100)
parser.add_argument("--loss_root", type=str, default='./experiments/loss')
parser.add_argument(
    "--adv_image_root", type=str, default='./experiments/adv_images')
parser.add_argument("--clean_image_root", type=str, default='data/images_1000')
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--arch", type=str, default='resnet34')
parser.add_argument(
    "--target_archs", type=str, default=['densenet201'], nargs='*')

parser.add_argument("--attack_method", type=str, default='PGD')
parser.add_argument("--gamma", type=float, default=1.)
parser.add_argument("--momentum", type=float, default=0.)
parser.add_argument("--m", type=int, default=0)
parser.add_argument("--sigma", type=float, default=15.)
parser.add_argument("--ti_size", type=int, default=1)
parser.add_argument("--lam", type=float, default=0.)
parser.add_argument("--grid_scale", type=int, default=16)
parser.add_argument("--sample_grid_num", type=int, default=32)
parser.add_argument("--sample_times", type=int, default=32)
args = parser.parse_args()

target_archs = [
    "vgg16", "resnet152", "densenet201", "senet154", "inceptionv3",
    "inceptionv4", "inceptionresnetv2"
]


def test_interaction_reduced_attack():
    set_config(args)
    interaction_reduced_attack.generate_adv_images(args)
    for target_arch in target_archs:
        args.target_arch = target_arch
        interaction_reduced_attack.save_scores(args)
        leave_one_out.evaluate(args)


test_interaction_reduced_attack()

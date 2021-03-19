import os

import numpy as np
from imageio import imsave


def save_images(images, filename, output_dir):
    cur_images = (np.round(images[0, :, :, :] * 255)).astype(np.uint8)
    with open(os.path.join(output_dir, filename), 'wb') as f:
        imsave(f, cur_images.transpose(1, 2, 0), format='png')


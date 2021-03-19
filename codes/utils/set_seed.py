import os
import random

import numpy as np
import torch


def seed_torch(seed=0):
    if isinstance(seed, str):
        seed = np.uint32(hash(seed))
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

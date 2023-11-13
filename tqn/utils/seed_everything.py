import os
import random

import numpy as np
import torch as T


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    T.use_deterministic_algorithms(True)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False

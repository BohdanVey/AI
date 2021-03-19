import os
import shutil
import numpy as np
import random
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_experiment_log_dir(exp_path, base_dir='exp_logs'):
    exp_path = os.path.join(base_dir, exp_path)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    else:
        while True:
            ans = input("Path `{}` exists, do you want to delete it?: [y/n]".format(exp_path))
            if ans == "y":
                shutil.rmtree(exp_path)
                os.makedirs(exp_path)
                break
            elif ans == "n":
                return None
    logdir_path = os.path.join(exp_path, 'logdir')
    os.makedirs(logdir_path, exist_ok=True)
    return logdir_path
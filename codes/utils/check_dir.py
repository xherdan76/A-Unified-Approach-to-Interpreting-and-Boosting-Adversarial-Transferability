import os
import shutil


def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def rm_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def reset_dir(path):
    rm_dir(path)
    mk_dir(path)

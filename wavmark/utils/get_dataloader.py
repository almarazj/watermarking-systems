from torch.utils.data import DataLoader
from typing import Dict
import os
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import random


def get_loader(seed: int, config: dict) -> Dict[str, DataLoader]:
    """
    Creates PyTorch DataLoaders for training, development, and evaluation datasets.

    Parameters
    ----------
    seed : int
        The seed for random number generation to ensure reproducibility.
    config : dict
        A dictionary containing paths and parameters required for creating DataLoaders.

    Returns
    -------
    dict
        A dictionary containing DataLoaders with keys 'train', 'dev', and 'eval'.
    """

    loaders = {}

    # Paths and protocol files
    trn_set_path = config.get("train_set_path")
    dev_set_path = config.get("dev_set_path")
    eval_set_path = config.get("eval_set_path")
    trn_list_path = config.get("train_set_protocol")
    dev_list_path = config.get("dev_set_protocol")
    eval_list_path = config.get("eval_set_protocol")

    # Training
    if trn_set_path and trn_list_path and os.path.exists(trn_set_path) and os.path.exists(trn_list_path):
        
        file_train = gen_spoof_list(dir_meta=trn_list_path)
        print("no. training files:", len(file_train))

        train_set = GetDataset(list_IDs=file_train, base_dir=trn_set_path)

        gen = torch.Generator()
        gen.manual_seed(seed)

        trn_loader = DataLoader(train_set,
                                batch_size=config.get("batch_size", 24),
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                worker_init_fn=seed_worker,
                                generator=gen)
        
        loaders["train"] = trn_loader

    # Validation
    if dev_set_path and dev_list_path and os.path.exists(dev_set_path) and os.path.exists(dev_list_path):
        _, file_dev = gen_spoof_list(dir_meta=dev_list_path)
        print("no. validation files:", len(file_dev))

        dev_set = GetDataset(list_IDs=file_dev, base_dir=dev_set_path)

        dev_loader = DataLoader(dev_set,
                                batch_size=config.get("batch_size", 24),
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)

        loaders["dev"] = dev_loader

    # Evaluation
    if eval_set_path and eval_list_path and os.path.exists(eval_set_path) and os.path.exists(eval_list_path):
        file_eval = gen_spoof_list(dir_meta=eval_list_path)
        print("no. evaluation files:", len(file_eval))

        eval_set = GetDataset(list_IDs=file_eval, base_dir=eval_set_path)

        eval_loader = DataLoader(eval_set,
                                 batch_size=config.get("batch_size", 24),
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=True)

        loaders["eval"] = eval_loader

    return loaders

class GetDataset(Dataset):
    def __init__(self, list_IDs, base_dir):
        """
        self.list_IDs : list of strings (each string: utt key)
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 16000  # take 1 sec audio

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        file_path = self.base_dir / f"{key}.wav"
        x,  = sf.read(file_path)
        x_pad = pad_random(x, self.cut)
        x_inp = Tensor(x_pad)
        return x_inp
        
def gen_spoof_list(dir):
    """
    Read protocols file and generate a list containing filenames of spoofed data.
    """
    file_list = []
    with open(dir, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        _, key, _, label = line.strip().split(" ")
        if label == "spoof":
            file_list.append(key)
            
    return file_list

def pad_random(x: np.ndarray, max_len: int = 16000):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(0, x_len - max_len + 1)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(np.ceil(max_len / x_len))
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x
    
def seed_worker(worker_id):
    """
    Used in generating seed for the worker of torch.utils.data.Dataloader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
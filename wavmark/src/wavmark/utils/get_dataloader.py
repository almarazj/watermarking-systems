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
    dev_trial_path = config.get("dev_set_protocol")
    eval_trial_path = config.get("eval_set_protocol")

    # Training
    if trn_set_path and trn_list_path and os.path.exists(trn_set_path) and os.path.exists(trn_list_path):
        d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                                is_train=True,
                                                is_eval=False)
        print("no. training files:", len(file_train))

        train_set = GetDataset(list_IDs=file_train,
                                  labels=d_label_trn,
                                  base_dir=trn_set_path)

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
    if dev_set_path and dev_trial_path and os.path.exists(dev_set_path) and os.path.exists(dev_trial_path):
        _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                    is_train=False,
                                    is_eval=False)
        print("no. validation files:", len(file_dev))

        dev_set = GetDataset(list_IDs=file_dev,
                                   base_dir=dev_set_path)

        dev_loader = DataLoader(dev_set,
                                batch_size=config.get("batch_size", 24),
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)

        loaders["dev"] = dev_loader

    # Evaluation
    if eval_set_path and eval_trial_path and os.path.exists(eval_set_path) and os.path.exists(eval_trial_path):
        file_eval = genSpoof_list(dir_meta=eval_trial_path,
                                  is_train=False,
                                  is_eval=True)
        print("no. evaluation files:", len(file_eval))

        eval_set = GetDataset(list_IDs=file_eval,
                                    base_dir=eval_set_path)

        eval_loader = DataLoader(eval_set,
                                 batch_size=config.get("batch_size", 24),
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=True)

        loaders["eval"] = eval_loader

    return loaders

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(0, x_len - max_len + 1)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(np.ceil(max_len / x_len))
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


class GetDataset(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        file_path = os.path.join(self.base_dir, f"{key}.wav")
        X, _ = sf.read(file_path)
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y
    
def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list
    
def seed_worker(worker_id):
    """
    Used in generating seed for the worker of torch.utils.data.Dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed):
    """ 
    set initial seed for reproduction
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

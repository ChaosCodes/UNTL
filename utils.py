import torch
import tqdm
import numpy as np
import random
import logging


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def save_model(model, path):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)


def save_adapter_model(model, adapter, path):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'adapter_model_state_dict': adapter.state_dict(),
    }, path)



class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record) 


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(dataset1)
        self.len2 = len(dataset2)

    def __getitem__(self, idx):
        item = {
            'input_ids_1': self.dataset1[idx % self.len1]['input_ids'],
            'attention_mask_1': self.dataset1[idx % self.len1]['attention_mask'],
            'label_1': self.dataset1[idx % self.len1]['label'],
            'input_ids_2': self.dataset2[idx % self.len2]['input_ids'],
            'attention_mask_2': self.dataset2[idx % self.len2]['attention_mask'],
            'label_2': self.dataset2[idx % self.len2]['label']
        }
        return item

    def __len__(self):
        return max(self.len1, self.len2)


class TripletCustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2, dataset3):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.len1 = len(dataset1)
        self.len2 = len(dataset2)
        self.len3 = len(dataset3)

    def __getitem__(self, idx):
        item = {
            'input_ids_1': self.dataset1[idx % self.len1]['input_ids'],
            'attention_mask_1': self.dataset1[idx % self.len1]['attention_mask'],
            'label_1': self.dataset1[idx % self.len1]['label'],
            'input_ids_2': self.dataset2[idx % self.len2]['input_ids'],
            'attention_mask_2': self.dataset2[idx % self.len2]['attention_mask'],
            'label_2': self.dataset2[idx % self.len2]['label'],
            'input_ids_3': self.dataset3[idx % self.len3]['input_ids'],
            'attention_mask_3': self.dataset3[idx % self.len3]['attention_mask'],
            'label_3': self.dataset3[idx % self.len3]['label']
        }
        return item

    def __len__(self):
        return max(self.len1, self.len2, self.len3)

import torch
from torch.utils.data import Dataset

class CWRUDataset(Dataset):
    def __init__(self, path):
        self.dataset = torch.load(path)
        self.data    = self.dataset['data']
        self.label   = self.dataset['label']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index].float() / 255.0
        label = self.label[index].long()
        return image, label
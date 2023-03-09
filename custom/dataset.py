from torch.utils.data import Dataset, DataLoader
import torch

class SegmentationDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class SegmentationDataloader(DataLoader):
    def __init__(self, x, y, batch_size):
        dataset = SegmentationDataset(x, y)
        super().__init__(dataset, batch_size=batch_size, shuffle=True)

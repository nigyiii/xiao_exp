import torch
from torch.utils.data import Dataset

class XiaoDataset(Dataset):
    """定义数据集"""

    def __init__(self, txt_file):
        """
        txt_file（string）：txt文件的路径。
        """
        self.data_list = [list(map(eval, line.strip().split())) for line in open(txt_file)]


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = {'input': torch.Tensor(self.data_list[idx][:3]), 'output': torch.Tensor(self.data_list[idx][3:])}
        return sample
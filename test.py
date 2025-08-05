import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from brats.dataset import get_seg_dataset
import torchvision

if __name__ == "__main__":
    _, _, test_dataset = get_seg_dataset(
            "brats_output_path",
        )

    test_loader = DataLoader(
            test_dataset,
            batch_size=12,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=4,
        )
    ti=next(iter(test_loader))["seg"][11]

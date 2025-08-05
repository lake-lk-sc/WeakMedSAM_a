from torch.utils.data import DataLoader
from tqdm import tqdm
from brats.dataset import get_seg_dataset

if __name__ == "__main__":
    _, _, test_dataset = get_seg_dataset(
            "brats_output_path",
        )

    test_loader = DataLoader(
            test_dataset,
            batch_size=24,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=4,
        )

    for datapack in tqdm(test_loader, total=len(test_loader)/10):
        print(datapack["img"].shape)
        print(datapack["seg"].shape)
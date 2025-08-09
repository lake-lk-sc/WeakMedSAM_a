import argparse
import csv
import os

import SimpleITK as sitk
import numpy
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from brats.dataset import no_aug
from unet import UNet


class GrayscaleSliceDataset(Dataset):
    """用于加载单个病人所有灰度切片的数据集"""

    def __init__(self, slice_files):
        slice_files.sort(key=lambda f: int(os.path.basename(f).split('-')[-1].split('.')[0]))
        self.slice_files = slice_files

    def __len__(self):
        return len(self.slice_files)

    def __getitem__(self, index):
        # 以灰度模式 'L' 打开图像，并进行预处理
        img = Image.open(self.slice_files[index]).convert('L')
        img, _ = no_aug(img, img)
        return img


def generate_segmentation_volume_from_slices(model, device, slice_files, batch_size):
    """从灰度图文件列表高效生成一个3D的二值分割体。"""
    print(f"开始对 {len(slice_files)} 个切片进行推理...")
    dataset = GrayscaleSliceDataset(slice_files)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_pred_slices = []
    model.eval()
    with torch.no_grad():
        for image_batch in tqdm(loader, desc="模型推理中"):
            image_batch = image_batch.to(device)
            logits = model(image_batch)
            pred_map = torch.argmax(logits, dim=1)
            all_pred_slices.append(pred_map.cpu())

    full_volume_tensor = torch.cat(all_pred_slices, dim=0)
    return full_volume_tensor.numpy()


def create_and_save_point_cloud(volume_3d, spacing, down_sample_num: int, output_filename):
    """从3D的二值分割体生成点云并保存。"""
    print(f"正在为 {os.path.basename(output_filename)} 生成点云...")

    point_coordinates = np.argwhere(volume_3d == 1)

    if point_coordinates.shape[0] == 0:
        print(f"警告: 在 {os.path.basename(output_filename)} 中未找到前景点，跳过点云生成。")
        return

    # 将像素坐标转换为物理世界坐标（mm）
    spacing_vector = np.array([spacing[2], spacing[1], spacing[0]])  # 对应 [z, y, x] 的间距
    point_coordinates_mm = numpy.multiply(point_coordinates, spacing_vector)

    print(f"找到了 {point_coordinates.shape[0]} 个前景点。")

    pcd = o3d.geometry.PointCloud()
    # Open3D 期望的坐标顺序是 (x, y, z)
    pcd.points = o3d.utility.Vector3dVector(point_coordinates_mm[:, [2, 1, 0]])

    if down_sample_num > 0 and len(pcd.points) > down_sample_num:
        print(f"正在下采样到 {down_sample_num} 个点...")
        pcd = pcd.farthest_point_down_sample(down_sample_num)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    o3d.io.write_point_cloud(output_filename, pcd)
    print(f"点云已保存到: {output_filename}")


def get_nifti_spacing(nifti_file_path):
    """从一个NIfTI文件中读取体素间距(spacing)信息"""
    if not nifti_file_path:
        return np.array([1.0, 1.0, 1.0])
    try:
        nifti_image = sitk.ReadImage(nifti_file_path)
        spacing = nifti_image.GetSpacing()
        print(f"成功读取Spacing信息: (x, y, z) = {spacing} mm")
        return np.array(spacing)
    except Exception as e:
        print(f"错误：无法读取NIfTI文件 {nifti_file_path}。错误信息: {e}")
        return np.array([1.0, 1.0, 1.0])


def load_patient_grade_mapping(mapping_csv_path):
    """从 name_mapping.csv 加载 patient ID 到 Grade (HGG/LGG) 的映射。"""
    mapping = {}
    try:
        with open(mapping_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row['BraTS_2019_subject_ID'].strip()
                grade = row['Grade'].strip()
                if patient_id and grade:
                    mapping[patient_id] = grade
        print(f"成功从 {mapping_csv_path} 加载了 {len(mapping)} 个病人ID映射。")
        return mapping
    except FileNotFoundError:
        print(f"错误: 找不到映射文件 {mapping_csv_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description="批量处理BraTS病人数据，生成3D分割并保存为点云文件。")
    parser.add_argument("--input_dir", type=str, default="brats_output_path", help="包含所有病人灰度图子文件夹的根目录。")
    parser.add_argument("--output_dir", type=str, default="point_cloud_output_path", help="保存生成的.ply点云文件的目录。")
    parser.add_argument("--model_path", type=str, default="logdir_u/run001/run001.pth", help="训练好的U-Net模型文件路径。")
    parser.add_argument("--nifti_root", type=str, default="brats_input_path/MICCAI_BraTS_2019_Data_Training", help="包含原始NIfTI文件的根目录 (用于获取spacing)。")
    parser.add_argument("--mapping_csv", type=str, default="brats_input_path/MICCAI_BraTS_2019_Data_Training/name_mapping.csv", help="病人ID到HGG/LGG的映射CSV文件。")
    parser.add_argument("--batch_size", type=int, default=32, help="模型推理时的批处理大小。")
    parser.add_argument("--down_sample", type=int, default=4096, help="点云下采样的点数。设置为0则不进行下采样。")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载模型
    model = UNet(n_channels=1, n_classes=2, bilinear=True)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # 2. 加载病人ID到Grade的映射
    grade_mapping = load_patient_grade_mapping(args.mapping_csv)
    if not grade_mapping:
        return

    # 3. 获取所有病人文件夹
    try:
        patient_ids = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    except FileNotFoundError:
        print(f"错误: 输入目录 {args.input_dir} 不存在。")
        return

    print(f"在 {args.input_dir} 中找到 {len(patient_ids)} 个病人。")

    # 4. 遍历处理每个病人
    for patient_id in tqdm(patient_ids, desc="处理所有病人"):
        print(f"\n--- 开始处理病人: {patient_id} ---")

        patient_grayscale_dir = os.path.join(args.input_dir, patient_id)
        output_ply_path = os.path.join(args.output_dir, f"{patient_id}.ply")

        if os.path.exists(output_ply_path):
            print(f"文件 {output_ply_path} 已存在，跳过。")
            continue

        grade = grade_mapping.get(patient_id)
        nifti_path = None
        if not grade:
            print(f"警告: 在mapping.csv中找不到病人 {patient_id} 的分级(Grade)信息。")
        else:
            # 尝试查找 .nii 和 .nii.gz 文件
            for ext in ["_seg.nii", "_seg.nii.gz"]:
                path = os.path.join(args.nifti_root, grade, patient_id, f"{patient_id}{ext}")
                if os.path.exists(path):
                    nifti_path = path
                    break
            if not nifti_path:
                 print(f"警告: 找不到病人 {patient_id} 的NIfTI文件。")

        spacing = get_nifti_spacing(nifti_path)

        slice_files = [os.path.join(patient_grayscale_dir, f) for f in os.listdir(patient_grayscale_dir) if f.endswith('.jpg')]
        if not slice_files:
            print(f"错误：在文件夹 {patient_grayscale_dir} 中找不到任何图像文件！")
            continue

        segmentation_volume = generate_segmentation_volume_from_slices(model, device, slice_files, args.batch_size)
        create_and_save_point_cloud(segmentation_volume, spacing, args.down_sample, output_ply_path)

    print("\n所有病人处理完成！")


if __name__ == "__main__":
    main()
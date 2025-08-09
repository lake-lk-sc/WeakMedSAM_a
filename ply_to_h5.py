import os
import argparse
import h5py
import numpy as np
import open3d as o3d
from tqdm import tqdm

def get_args():
    """解析和返回命令行参数"""
    parser = argparse.ArgumentParser(
        description="Convert a directory of .ply point cloud files into a single HDF5 dataset."
    )

    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="Directory where the point cloud files (.ply) are stored."
    )

    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True, 
        help="Path to the output HDF5 file (e.g., 'point_cloud_dataset.h5')."
    )

    return parser.parse_args()

def create_hdf5_dataset(input_dir: str, output_file: str):
    """将PLY文件目录转换为HDF5数据集"""
    ply_files = [f for f in os.listdir(input_dir) if f.endswith('.ply')]

    if not ply_files:
        print(f"错误: 在目录 '{input_dir}' 中没有找到 .ply 文件。")
        return

    print(f"找到 {len(ply_files)} 个 .ply 文件。正在开始转换...")

    with h5py.File(output_file, 'w') as hf:
        for ply_file in tqdm(ply_files, desc="Processing files"):
            patient_id = os.path.splitext(ply_file)[0]
            file_path = os.path.join(input_dir, ply_file)

            try:
                pcd = o3d.io.read_point_cloud(file_path)
                if not pcd.has_points():
                    tqdm.write(f"警告: 文件 {ply_file} 没有点，已跳过。")
                    continue

                points = np.asarray(pcd.points)

                # 在HDF5文件中为每个病人创建一个组
                patient_group = hf.create_group(patient_id)
                # 将点云数据存储在组内
                patient_group.create_dataset('points', data=points, compression="gzip")

            except Exception as e:
                tqdm.write(f"处理文件 {ply_file} 时出错: {e}")

    print(f"\n数据集成功创建于: {output_file}")

if __name__ == "__main__":
    args = get_args()
    create_hdf5_dataset(args.input_dir, args.output_file)

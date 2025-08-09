import os
import open3d as o3d
from PIL import Image
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from typing import Optional
import argparse

def get_args():
    """解析和返回命令行参数"""
    parser = argparse.ArgumentParser(description="Visualize medical imaging data for a specific patient.")
    
    parser.add_argument("--patient_id", type=str, required=True, 
                        help="The ID of the patient to visualize (e.g., 'BraTS19_2013_0_1').")
    
    parser.add_argument("--slice_index", type=int, required=True, 
                        help="The slice index to display for 2D images.")
    
    parser.add_argument("--point_cloud_dir", type=str, default="point_cloud_output_path", 
                        help="Directory where the point cloud files (.ply) are stored.")
                        
    parser.add_argument("--grayscale_dir", type=str, default="brats_output_path", 
                        help="Directory where the grayscale slice images are stored.")
                        
    parser.add_argument("--nifti_base_dir", type=str, default="brats_input_path/MICCAI_BraTS_2019_Data_Training",
                        help="Base directory to search for NIfTI files recursively.")
                        
    parser.add_argument("--nifti_modality", type=str, default="_seg.nii", 
                        help="Modality of the NIfTI file to visualize (e.g., '_seg.nii', '_t1.nii').")
                        
    parser.add_argument("--pseudo_label_dir", type=str, default="pseudo_label_path", 
                        help="Directory where the pseudo label images are stored.")

    parser.add_argument("--save_figure", action='store_true', 
                        help="If set, the final 2D visualization will be saved to a file.")
                        
    parser.add_argument("--save_output_dir", type=str, default="visualize_output", 
                        help="Directory to save the output visualization figure.")

    return parser.parse_args()


def visualize_point_cloud(patient_id: str, base_dir: str):
    """从指定目录加载并可视化病人的点云文件 (.ply)"""
    ply_file = os.path.join(base_dir, f"{patient_id}.ply")
    print(f"正在加载点云文件: {ply_file}")

    if not os.path.exists(ply_file):
        print(f"错误: 点云文件未找到: {ply_file}")
        return

    try:
        pcd = o3d.io.read_point_cloud(ply_file)
        if not pcd.has_points():
            print("警告: 点云文件中没有点。")
            return

        print("正在启动点云可视化窗口... (请关闭窗口后程序才会继续)")
        mesh_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=50, origin=pcd.get_center() - np.array([50, 50, 50])
        )
        o3d.visualization.draw_geometries([pcd, mesh_coordinate_frame])
    except Exception as e:
        print(f"可视化点云时发生错误: {e}")


def visualize_grayscale_slice(patient_id: str, base_dir: str, slice_index: int, ax: Optional[plt.Axes] = None):
    """可视化指定病人的灰度图切片"""
    patient_dir = os.path.join(base_dir, patient_id)
    title = f"Grayscale Slice #{slice_index}"
    print(f"正在加载灰度图: {patient_dir}")

    if not os.path.isdir(patient_dir):
        print(f"错误: 找不到目录: {patient_dir}")
        if ax:
            ax.text(0.5, 0.5, 'Directory not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.axis('off')
        return

    slice_files = [os.path.join(patient_dir, f) for f in os.listdir(patient_dir) if f.endswith(('.jpg', '.png'))]
    if not slice_files:
        print(f"错误: 在 {patient_dir} 中找不到图像文件。")
        if ax:
            ax.text(0.5, 0.5, 'Images not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.axis('off')
        return

    try:
        slice_files.sort(key=lambda f: int(os.path.basename(f).split('-')[-1].split('.')[0]))
    except (IndexError, ValueError):
        print("警告: 无法根据 'slice-NUMBER.ext' 格式排序文件名，将使用字母顺序排序。")
        slice_files.sort()

    if slice_index >= len(slice_files):
        print(f"错误: 切片索引 {slice_index} 超出范围 (共 {len(slice_files)} 个切片)。")
        if ax:
            ax.text(0.5, 0.5, 'Index out of range', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.axis('off')
        return

    img_path = slice_files[slice_index]
    try:
        img = Image.open(img_path).convert('L')
        show_plot = ax is None
        if show_plot:
            fig, ax = plt.subplots()

        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

        if show_plot:
            plt.show()

    except Exception as e:
        print(f"打开或显示图像时出错 {img_path}: {e}")


def find_patient_nifti_path(base_dir: str, patient_id: str, modality_suffix: str) -> Optional[str]:
    """在基础目录中递归搜索病人的NIfTI文件"""
    for root, dirs, _ in os.walk(base_dir):
        if patient_id in dirs:
            patient_dir = os.path.join(root, patient_id)
            for ext in [modality_suffix, modality_suffix + '.gz']:
                nifti_file_name = f"{patient_id}{ext}"
                nifti_file_path = os.path.join(patient_dir, nifti_file_name)
                if os.path.exists(nifti_file_path):
                    return nifti_file_path
    return None


def visualize_nifti_slice(patient_id: str, base_dir: str, modality: str, slice_index: int, ax: Optional[plt.Axes] = None):
    """可视化指定病人的NIfTI文件切片"""
    title = f"NIfTI Slice #{slice_index}\n({modality.replace('.nii','')})"
    print(f"正在搜索NIfTI文件: {patient_id}{modality} in {base_dir}")

    nifti_path = find_patient_nifti_path(base_dir, patient_id, modality)

    if not nifti_path:
        print(f"错误: 找不到NIfTI文件。")
        if ax:
            ax.text(0.5, 0.5, 'NIfTI file not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.axis('off')
        return

    print(f"找到并加载: {nifti_path}")
    try:
        nifti_image = sitk.ReadImage(nifti_path)
        image_array = sitk.GetArrayFromImage(nifti_image)

        if slice_index >= image_array.shape[0]:
            print(f"错误: 切片索引 {slice_index} 超出范围 (Z轴大小为 {image_array.shape[0]})。")
            if ax:
                ax.text(0.5, 0.5, 'Index out of range', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                ax.axis('off')
            return

        slice_data = image_array[slice_index, :, :]

        show_plot = ax is None
        if show_plot:
            fig, ax = plt.subplots()

        ax.imshow(slice_data, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

        if show_plot:
            plt.show()

    except Exception as e:
        print(f"读取或显示NIfTI文件时出错 {nifti_path}: {e}")


def visualize_pseudo_label_slice(patient_id: str, base_dir: str, slice_index: int, ax: Optional[plt.Axes] = None):
    """可视化指定病人的伪标签切片"""
    patient_dir = os.path.join(base_dir, patient_id)
    title = f"Pseudo Label Slice #{slice_index}"
    print(f"正在加载伪标签: {patient_dir}")

    if not os.path.isdir(patient_dir):
        print(f"错误: 找不到目录: {patient_dir}")
        if ax:
            ax.text(0.5, 0.5, 'Directory not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.axis('off')
        return

    slice_files = [os.path.join(patient_dir, f) for f in os.listdir(patient_dir) if f.endswith(('.jpg', '.png'))]
    if not slice_files:
        print(f"错误: 在 {patient_dir} 中找不到图像文件。")
        if ax:
            ax.text(0.5, 0.5, 'Images not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.axis('off')
        return

    try:
        slice_files.sort(key=lambda f: int(os.path.basename(f).split('-')[-1].split('.')[0]))
    except (IndexError, ValueError):
        print("警告: 无法根据数字后缀排序文件名，将使用字母顺序排序。")
        slice_files.sort()

    if slice_index >= len(slice_files):
        print(f"错误: 切片索引 {slice_index} 超出范围 (共 {len(slice_files)} 个切片)。")
        if ax:
            ax.text(0.5, 0.5, 'Index out of range', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.axis('off')
        return

    img_path = slice_files[slice_index]
    try:
        img = Image.open(img_path)
        show_plot = ax is None
        if show_plot:
            fig, ax = plt.subplots()

        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

        if show_plot:
            plt.show()

    except Exception as e:
        print(f"打开或显示图像时出错 {img_path}: {e}")


if __name__ == "__main__":
    args = get_args()

    # 第一部分: 可视化3D点云
    visualize_point_cloud(args.patient_id, args.point_cloud_dir)

    # 第二部分: 可视化2D切片
    print("\n正在准备2D切片可视化...")

    plot_configs = [
        {"func": visualize_grayscale_slice, "args": (args.patient_id, args.grayscale_dir, args.slice_index), "path": os.path.join(args.grayscale_dir, args.patient_id)},
        {"func": visualize_nifti_slice, "args": (args.patient_id, args.nifti_base_dir, args.nifti_modality, args.slice_index), "path": find_patient_nifti_path(args.nifti_base_dir, args.patient_id, args.nifti_modality)},
        {"func": visualize_pseudo_label_slice, "args": (args.patient_id, args.pseudo_label_dir, args.slice_index), "path": os.path.join(args.pseudo_label_dir, args.patient_id)}
    ]

    valid_plots = [p for p in plot_configs if p["path"] and os.path.exists(p["path"])]

    if valid_plots:
        num_plots = len(valid_plots)
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6.5))
        fig.suptitle(f"Patient: {args.patient_id}, Slice: {args.slice_index}", fontsize=16)

        if num_plots == 1:
            axes = [axes]

        for i, plot_info in enumerate(valid_plots):
            plot_info["func"](*plot_info["args"], ax=axes[i])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if args.save_figure:
            os.makedirs(args.save_output_dir, exist_ok=True)
            save_path = os.path.join(args.save_output_dir, f"comparison_{args.patient_id}_slice_{args.slice_index}.png")
            print(f"\n正在保存2D切片组合图像到: {save_path}")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print("保存成功！")

        print("\n正在显示2D切片图像... (关闭窗口后程序将结束)")
        plt.show()
    else:
        print("\n没有可供可视化的2D切片数据。请检查路径配置和病人ID。")

    print("\n可视化任务完成！")

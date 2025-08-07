import numpy
import torch
import os
import open3d as o3d
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
from unet import UNet
import SimpleITK as sitk
from brats.dataset import no_aug

# ===================================================================
#   第一部分: 数据加载与GPU推理
# ===================================================================

class GrayscaleSliceDataset(Dataset):
    """一个专门加载灰度图切片序列的Dataset"""

    def __init__(self, slice_files):
        # 按文件名排序，保证Z轴（切片）顺序正确
        slice_files.sort(key=lambda f: int(os.path.basename(f).split('-')[-1].split('.')[0]))
        self.slice_files = slice_files
        # 定义一个简单的变换：直接转为张量 (因为已经是256x256灰度图了)
    def __len__(self):
        return len(self.slice_files)

    def __getitem__(self, index):
        # 以灰度模式 'L' 打开图像
        img = Image.open(self.slice_files[index]).convert('L')
        img,_ = no_aug(img,img)
        return img


def generate_segmentation_volume_from_slices(model, device, slice_files, batch_size=16):
    """
    接收一个灰度图文件列表，高效生成一个3D的二值分割体。
    返回一个 [Z, Y, X] 形状的Numpy数组，值为0或1。
    """
    print(f"开始推理，总共 {len(slice_files)} 个切片...")

    dataset = GrayscaleSliceDataset(slice_files)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_pred_slices = []

    model.eval()
    with torch.no_grad():
        for image_batch in tqdm(loader, desc="模型推理中"):
            image_batch = image_batch.to(device)
            logits = model(image_batch)
            pred_map = torch.argmax(logits, dim=1)  # 从Logits到二值图
            all_pred_slices.append(pred_map.cpu())

    # 将所有批次的预测结果拼接成一个大的3D张量
    full_volume_tensor = torch.cat(all_pred_slices, dim=0)

    return full_volume_tensor.numpy()



def create_and_show_point_cloud(volume_3d,spacing, output_filename=None):
    """
    从3D的二值分割体生成点云，并进行保存和可视化。
    """
    print("正在从3D分割图中生成点云...")

    point_coordinates = np.argwhere(volume_3d == 1)

    spacing_vector = np.array([spacing[2], spacing[1], spacing[0]])  # 对应 [z, y, x] 的间距

    point_coordinates_mm = numpy.multiply(point_coordinates, spacing_vector)
    if point_coordinates.shape[0] == 0:
        print("警告: 未在分割图中找到任何前景点，无法生成点云。")
        return

    print(f"找到了 {point_coordinates.shape[0]} 个前景点。")

    # 2. 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(point_coordinates_mm[:, [2, 1, 0]])

    pcd.paint_uniform_color([0.2, 1.0, 1.0])
    min_bounds_mm=pcd.get_min_bound()

    volume_shape_pixels = volume_3d.shape
    max_bounds_mm = numpy.multiply((np.array(volume_shape_pixels[::-1])), spacing_vector)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bounds_mm, max_bound=max_bounds_mm)
    bbox.color = (0.5, 0.5, 0.5)

    mesh_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=50, origin=[0, 0, 0]
    )
    mesh_coordinate_frame.translate(min_bounds_mm)

    if output_filename:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        o3d.io.write_point_cloud(output_filename, pcd)
        print(f"点云已保存到: {output_filename}")

    # 5. 可视化点云
    print("正在启动点云可视化窗口... (请关闭窗口后程序才会结束)")
    o3d.visualization.draw_geometries([pcd,bbox, mesh_coordinate_frame])


def get_nifti_spacing(nifti_file_path):
    """从一个NIfTI文件中读取体素间距(spacing)信息"""
    try:
        nifti_image = sitk.ReadImage(nifti_file_path)
        # GetSpacing() 返回一个元组 (spacing_x, spacing_y, spacing_z)
        spacing = nifti_image.GetSpacing()
        print(f"成功读取Spacing信息: (x, y, z) = {spacing} mm")
        return np.array(spacing)
    except Exception as e:
        print(f"错误：无法读取NIfTI文件 {nifti_file_path}。错误信息: {e}")
        # 如果失败，返回一个默认值，点云将以像素为单位
        return np.array([1.0, 1.0, 1.0])

if __name__ == "__main__":
    # --- 用户需要配置的参数 ---

    # 1. 你要处理的那个病人的灰度图文件夹路径
    PATIENT_GRAYSCALE_DIR = "brats_output_path\\BraTS19_2013_0_1"

    # 2. 你训练好的U-Net模型文件路径
    MODEL_PATH = "logdir_u\\run001\\run001.pth"

    # 3. (可选) 你想保存点云文件的路径和名称
    POINT_CLOUD_OUTPUT_PATH = "point_cloud_output_path\\BraTS19_2013_0_1.ply"

    NIFTI_PATH="brats_input_path\\MICCAI_BraTS_2019_Data_Training\\LGG\\BraTS19_2013_0_1\\BraTS19_2013_0_1_seg.nii"

    BATCH_SIZE = 62

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型并移动到GPU
    model = UNet(n_channels=1, n_classes=2, bilinear=True)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    # 收集该病人的所有灰度图文件
    slice_files = [os.path.join(PATIENT_GRAYSCALE_DIR, f) for f in os.listdir(PATIENT_GRAYSCALE_DIR) if
                   f.endswith('.jpg')]

    if not slice_files:
        print(f"错误：在文件夹 {PATIENT_GRAYSCALE_DIR} 中找不到任何图像文件！")
    else:
        # **第一步：从灰度图序列生成3D二值分割体**
        segmentation_volume = generate_segmentation_volume_from_slices(model, device, slice_files, BATCH_SIZE)
        spacing = get_nifti_spacing(NIFTI_PATH)
        create_and_show_point_cloud(segmentation_volume,spacing,POINT_CLOUD_OUTPUT_PATH)

        print("\n任务完成！")
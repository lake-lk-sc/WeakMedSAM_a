# WeakMedSAM: 用于3D医学图像分割的弱监督学习框架

本项目是 WeakMedSAM 的官方实现，一个用于3D医学图像分割的弱监督学习框架。

## 环境准备

首先，克隆本仓库并进入项目目录。然后，使用以下命令安装所需的依赖包：

```bash
pip install -r requirements.txt
```

依赖包列表:
- SimpleITK
- scikit-learn
- scipy
- torch
- torchvision
- open3d
- pillow
- numpy
- tqdm
- matplotlib
- h5py

## 工作流程

本项目的核心思想是利用有限的监督信息（例如，图像级别的标签）来训练一个强大的分割模型。整个工作流程分为以下几个关键步骤：

1.  **数据准备**: 下载并预处理 BraTS 2019 数据集。
2.  **预聚类**: 对数据进行特征聚类，为后续的弱监督学习做准备。
3.  **训练 WeakMedSAM**: 使用聚类结果和图像级标签训练核心的 WeakMedSAM 模型。
4.  **生成伪标签**: 利用训练好的 WeakMedSAM 模型为无标签数据生成高质量的像素级伪标签。
5.  **训练分割网络**: 使用生成的伪标签来监督训练一个标准的分割网络（如 UNet）。
6.  **模型评估**: 评估最终分割网络的性能。

### 1. 数据准备

首先，从 [Kaggle](https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019) 下载 BraTS 2019 数据集。

下载并解压后，运行以下脚本进行数据预处理。请将 `$brats_input_path` 替换为你的原始数据集路径，将 `$brats_output_path` 替换为你希望保存预处理结果的路径。

```bash
python brats/preprocess.py --input-path $brats_input_path --output-path $brats_output_path
```

### 2. 预聚类

运行以下命令进行预聚类。
- `--data_path`: 指定预处理后的数据路径。
- `--save_path`: 指定聚类结果的保存路径。
- `--gpus`: 指定使用的 GPU。

```bash
python cluster.py --batch_size 256 --data_path $brats_output_path --data_module brats --parent_classes 1 --child_classes 8 --save_path $cluster_file_path --gpus $gpus
```

### 3. 训练 WeakMedSAM

在训练 WeakMedSAM 之前，需要下载 SAM ViT-b 的预训练权重。可以从 [Meta AI 的官方链接](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) 下载。

然后，运行以下命令开始训练。
- `--sam_ckpt`: SAM 预训练权重的路径。
- `--cluster_file`: 第二步生成的聚类文件路径。
- `--logdir`: 训练日志和模型权重的保存目录。
- `$index`: 实验索引或名称。

```bash
python train.py --seed 42 --sam_ckpt $sam_checkpoint_path --lr '1e-4' --batch_size 12 --max_epochs 10 --val_iters 3000 --index $index --data_path $brats_output_path --data_module brats --parent_classes 1 --child_classes 8 --child_weight 0.5 --cluster_file $cluster_file_path/brats-8.bin --logdir $logdir_s --gpus $gpus
```

### 4. 生成伪标签

使用训练好的 WeakMedSAM 模型来生成伪标签。
- `--samus-ckpt`: 上一步训练保存的 WeakMedSAM 模型权重路径。
- `--save-path`: 伪标签的保存路径。

```bash
python lab_gen.py --batch-size 24 --data-path $brats_output_path --save-path $pseudo_label_path --data-module brats --parent-classes 1 --child-classes 8 --samus-ckpt $logdir_s/$index/$index.pth --sam-ckpt sam_vit_b_01ec64.pth --t 4 --beta 4 --threshold 0.5 --gpus $gpus
```

### 5. 训练分割网络 (UNet)

利用生成的伪标签来训练最终的分割网络。
- `--lab_path`: 上一步生成的伪标签路径。
- `--logdir`: 分割网络训练日志和权重的保存目录。

```bash
python train_unet.py --seed 42 --lr '1e-4' --batch_size 128 --max_epochs 10 --val_iters 500 --index $index --data_path $brats_output_path --lab_path $pseudo_label_path --data_module brats --num_classes 2 --logdir $logdir_u --gpus $gpus
```

### 6. 模型评估

最后，评估分割网络的性能。
- `--ckpt`: 上一步训练保存的 UNet 模型权重路径。

```bash
python eval.py --data_path $brats_output_path --data_module 'brats' --batch_size 128 --num_classes 2 --ckpt $logdir_u/$index/$index.pth --gpus $gpus
```

## 结果

评估脚本会输出以下指标：

```
dice:           79.69
jaccard:        74.06
assd:           5.57
hd95:           28.34
```

- **Dice Coefficient**: Dice相似系数，衡量分割结果与真实标签的重叠度。
- **Jaccard Index**: Jaccard指数（IoU），另一种衡量重叠度的指标。
- **ASSD (Average Symmetric Surface Distance)**: 平均对称表面距离，衡量分割边界的匹配程度。
- **HD95 (95% Hausdorff Distance)**: 95%豪斯多夫距离，衡量两个点集之间的最大不匹配程度，对离群点更鲁棒。

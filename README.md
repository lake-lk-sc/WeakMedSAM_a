# Data Preparing

1. Download BraTS 2019 dataset from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019)
2. Run:
```bash
python brats/preprocess.py --input-path $brats_input_path --output-path $brats_output_path
```

# Pre-clustering

Run:
```bash
python cluster.py --batch_size 256 --data_path $brats_output_path --data_module brats --parent_classes 1 --child_classes 8 --save_path $cluster_file_path --gpus $gpus
```

# Training WeakMedSAM

1. Download the checkpoint of SAM ViT-b from [metaAI](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
2. Run:
```bash
python train.py --seed 42 --sam_ckpt $sam_checkpoint_path --lr '1e-4' --batch_size 12 --max_epochs 10 --val_iters 3000 --index $index --data_path $brats_output_path --data_module brats --parent_classes 1 --child_classes 8 --child_weight 0.5 --cluster_file $cluster_file_path/brats-8.bin --logdir $logdir_s --gpus $gpus
```

# Generating Pseudo Labels

Run:
```bash
python lab_gen.py --batch-size 24 --data-path $brats_output_path --save-path $pseudo_label_path --data-module brats --parent-classes 1 --child-classes 8 --samus-ckpt $logdir_s/$index/$index.pth --sam-ckpt sam_vit_b_01ec64.pth --t 4 --beta 4 --threshold 0.5 --gpus $gpus
```

# Training Segmentation Network

Run:
```bash
python train_unet.py --seed 42 --lr '1e-4' --batch_size 128 --max_epochs 10 --val_iters 500 --index $index --data_path $brats_output_path --lab_path $pseudo_label_path --data_module brats --num_classes 2 --logdir $logdir_u --gpus $gpus
```

# Evaluation

Run:
```bash
python eval.py --data_path $brats_output_path --data_module 'brats' --batch_size 128 --num_classes 2 --ckpt $logdir_u/$index/$index.pth --gpus $gpus
```

Output is like:
```
dice:           79.69
jaccard:        74.06
assd:           5.57
hd95:           28.34
```
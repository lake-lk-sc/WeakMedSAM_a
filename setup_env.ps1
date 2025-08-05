$project_root = Get-Location
$brats_input_path = "$project_root\brats_input_data"
$brats_output_path = "$project_root\brats_output_path"
$cluster_file_path = "$project_root\cluster_file_path"
$sam_checkpoint_path = "$project_root\sam_vit_b_01ec64.pth"
$logdir_s = "$project_root\logdir_s"
$logdir_u = "$project_root\logdir_u"
$pseudo_label_path = "$project_root\pseudo_label_path"

# 参数变量
$gpus = "0"
$index = "run001"

# 检查并创建输出目录，以避免错误
New-Item -ItemType Directory -Path $brats_output_path -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path $cluster_file_path -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path $logdir -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path $pseudo_label_path -ErrorAction SilentlyContinue | Out-Null

Write-Host "变量定义完成。"
set -ex
source /home/omar/miniforge3/etc/profile.d/conda.sh
conda activate openfl
cd /home/omar/Documents/mine/INTEL/openfl/openfl-workspace/experimental/workflow/VisionFlow
python --name_or_path "facebook/dinov2-base" --use_peft --task classfication
python --name_or_path "google/vit-base-patch16-224" --use_peft --task classfication
python --name_or_path "microsoft/resnet-50" --use_peft --task classfication

python --non_iid --name_or_path "facebook/dinov2-base" --use_peft --task classfication
python --non_iid --name_or_path "google/vit-base-patch16-224" --use_peft --task classfication
python --non_iid --name_or_path "microsoft/resnet-50" --use_peft --task classfication

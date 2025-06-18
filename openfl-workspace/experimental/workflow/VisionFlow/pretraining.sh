set -ex
source /home/omar/miniforge3/etc/profile.d/conda.sh
conda activate openfl
cd /home/omar/Documents/mine/INTEL/openfl/openfl-workspace/experimental/workflow/VisionFlow
python workflow.py --non_iid --name_or_path "facebook/dinov2-base" --use_peft --task pretraining --percentage 1.0 --just_train
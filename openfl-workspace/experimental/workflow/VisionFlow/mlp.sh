set -ex
source /home/omar/miniforge3/etc/profile.d/conda.sh
conda activate openfl
cd /home/omar/Documents/mine/INTEL/openfl/openfl-workspace/experimental/workflow/VisionFlow
for percentage in 0.05 0.1 0.3 0.5 1.0; do
    python workflow.py --non_iid --name_or_path "facebook/dinov2-base" --use_peft --task classification --percentage $percentage --head "MLPHead" --just_train
    python workflow.py --non_iid --name_or_path "google/vit-base-patch16-224" --use_peft --task classification --percentage $percentage --head "MLPHead" --just_train
    python workflow.py --non_iid --name_or_path "facebook/dinov2-base" --use_peft --task classification --percentage $percentage --with_pretrained --head "MLPHead" --just_train
done
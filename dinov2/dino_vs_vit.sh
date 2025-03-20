set -ex

for percentage in 0.1 0.3 0.6 1.0; do
    python workflow.py --use_decoder_unet --use_dino --use_vit --patient_percentage $percentage #--fast
    python workflow.py --use_decoder_unet --use_dino --use_vit --lora --patient_percentage $percentage #--fast
    python workflow.py --use_dino --use_vit --patient_percentage $percentage #--fast
    python workflow.py --use_vit --patient_percentage $percentage #--fast
    python workflow.py --patient_percentage $percentage #--fast
done
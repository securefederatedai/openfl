set -ex
experiment="few_shot"
for percentage in 1 6 16; do
    python workflow.py --use_decoder_unet --use_dino --use_vit --patient_count $percentage --experiment $experiment #--fast
    python workflow.py --use_decoder_unet --use_dino --use_vit --lora --patient_count $percentage --experiment $experiment #--fast
    python workflow.py --use_dino --use_vit --patient_count $percentage --experiment $experiment #--fast
    python workflow.py --use_vit --patient_count $percentage --experiment $experiment #--fast
    python workflow.py --patient_count $percentage --experiment $experiment #--fast
done
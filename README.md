# MLS_PJ

First, git clone or download all the files under the same directory.

Recover the environment setting at conda environment with command:

conda env create -f environment_transformer.yml

And you may need to run pip install timm to install timm library if you don't have it.

Use following command to get clean datasets for training:

unzip clean_resized_images.zip

# Attack
The following command is for attack:

python our_attack.py --attack OurAlgorithm --gpu 0 --batch_size 1 --model_name vit_base_patch16_224 --filename_prefix yours --target 0

the argument --attack is to determine the attack method
Currently, we provide OurAlgorithm, OurAlgorithm_MI, OurAlgorithm_SGM, OurAlgorithm_SGM, OurAlgorithm_SGM_MI, OurAlgorithm_DI, OurAlgorithm_DI_MI, OurAlgorithm_TI, OurAlgorithm_TI_MI, OurAlgorithm_SIM, OurAlgorithm_SGM_SIM
model_name: white-box model name, vit_base_patch16_224, pit_b_224, cait_s24_224, visformer_small
filename_prefix: additional names for the output file
target: targeted attack or non-targeted attack, 0 or 1

Or you can simply run sh run_attacks.sh OurAlgorithm 0 to triger the attack on four base models together.

# Evaluate
The following command is for evaluate on ViTs:

sh run_evaluate.sh 0 model_{model_name}-method_{attack}-{filename_prefix}-{target} target
for example: !sh run_evaluate.sh 0 model_vit_base_patch16_224-method_OurAlgorithm-yours-0 0

The following command is for evaluate on CNNs:
sh run_cnn_evaluate.sh 0 model_{model_name}-method_{attack}-{filename_prefix}-{target} target
for example: !sh run_cnn_evaluate.sh 0 model_vit_base_patch16_224-method_OurAlgorithm-yours-0 0

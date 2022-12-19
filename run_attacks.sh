echo "Attack method: $1, target attack: $2"
python our_attacks.py --attack $1 --gpu 0 --batch_size 1 --model_name vit_base_patch16_224 --filename_prefix yours --target $2
python our_attacks.py --attack $1 --gpu 0 --batch_size 1 --model_name pit_b_224 --filename_prefix yours --target $2
python our_attacks.py --attack $1 --gpu 0 --batch_size 1 --model_name cait_s24_224 --filename_prefix yours --target $2
python our_attacks.py --attack $1 --gpu 0 --batch_size 1 --model_name visformer_small --filename_prefix yours --target $2
wait
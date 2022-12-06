echo "target attack: $1"
python our_attacks.py --attack OurAlgorithm --gpu 0 --batch_size 1 --model_name vit_base_patch16_224 --filename_prefix yours --target $1
python our_attacks.py --attack OurAlgorithm --gpu 0 --batch_size 1 --model_name deit_base_distilled_patch16_224 --filename_prefix yours --target $1
python our_attacks.py --attack OurAlgorithm --gpu 0 --batch_size 1 --model_name levit_256 --filename_prefix yours --target $1
python our_attacks.py --attack OurAlgorithm --gpu 0 --batch_size 1 --model_name pit_b_224 --filename_prefix yours --target $1
python our_attacks.py --attack OurAlgorithm --gpu 0 --batch_size 1 --model_name cait_s24_224 --filename_prefix yours --target $1
python our_attacks.py --attack OurAlgorithm --gpu 0 --batch_size 1 --model_name convit_base --filename_prefix yours --target $1
python our_attacks.py --attack OurAlgorithm --gpu 0 --batch_size 1 --model_name tnt_s_patch16_224 --filename_prefix yours --target $1
python our_attacks.py --attack OurAlgorithm --gpu 0 --batch_size 1 --model_name visformer_small --filename_prefix yours --target $1
wait
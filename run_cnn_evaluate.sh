echo "The GPU device: $1, The adv_path: $2, target: $3"
python evaluate.py --gpu $1 --adv_path $2 --target $3 --model_name inception_v3 --batch_size 10 &
python evaluate.py --gpu $1 --adv_path $2 --target $3 --model_name inception_v4 --batch_size 10 &
python evaluate.py --gpu $1 --adv_path $2 --target $3 --model_name inception_resnet_v2 --batch_size 10 &
python evaluate.py --gpu $1 --adv_path $2 --target $3 --model_name resnetv2_152x2_bit_teacher --batch_size 10 
python evaluate.py --gpu $1 --adv_path $2 --target $3 --model_name res2net101_26w_4s --batch_size 10 &
python evaluate.py --gpu $1 --adv_path $2 --target $3 --model_name adv_inception_v3 --batch_size 10 &
python evaluate.py --gpu $1 --adv_path $2 --target $3 --model_name ens_adv_inception_resnet_v2 --batch_size 10 
wait
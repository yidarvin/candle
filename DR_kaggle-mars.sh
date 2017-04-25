declare -r path_train=/home/dnr/Documents/data/DR_kaggle/training
#declare -r path_test=/home/dnr/Documents/data/DR_kaggle/testing
declare -r path_val=/home/dnr/Documents/data/DR_kaggle/validation
#declare -r path_inf=/home/dnr/Documents/data/DR_kaggle/inference

declare -r name=DR_kaggle

declare -r path_model=/home/dnr/modelState/$name.ckpt
declare -r path_log=/home/dnr/logs/$name.txt

CUDA_VISIBLE_DEVICES=2,3 python /home/dnr/FirstAid/train_CNNclassification.py --pTrain $path_train --pVal $path_val --pModel $path_model --pLog $path_log --name $name --bConf 1 --bKappa 1 --nClass 5 --nGPU 2 --bs 64 --ep 50000 --lr 0.01

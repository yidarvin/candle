declare -r path_train=/home/dnr/Documents/data/messidor/training_5chan
#declare -r path_test=/home/dnr/Documents/data/DR_kaggle/testing
declare -r path_val=/home/dnr/Documents/data/messidor/testing_5chan
#declare -r path_inf=/home/dnr/Documents/data/DR_kaggle/inference

declare -r name=DR_messidor_5chan

declare -r path_model=/home/dnr/modelState/$name.ckpt
declare -r path_log=/home/dnr/logs/$name.txt

CUDA_VISIBLE_DEVICES=0,1 python /home/dnr/FirstAid/train_CNNclassification.py --pTrain $path_train --pVal $path_val --pModel $path_model --pLog $path_log --name $name --bConf 1 --bKappa 1 --nClass 4 --nGPU 2 --bs 16 --ep 50000 --lr 0.001 --do 0.3 --bLo 1

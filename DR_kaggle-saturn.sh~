declare -r path_train=/home/dnr/Documents/data/DR_kaggle/training
#declare -r path_test=/home/dnr/Documents/data/DR_kaggle/testing
declare -r path_val=/home/dnr/Documents/data/DR_kaggle/validation
#declare -r path_inf=/home/dnr/Documents/data/DR_kaggle/inference

declare -r name=DR_kaggle

declare -r path_model=/home/dnr/modelState/$name.ckpt
declare -r path_log=/home/dnr/logs/$name.txt

python /home/dnr/FirstAid/train_CNNclassification.py --pTrain $path_train --pVal $path_val --pModel $path_model --pLog $path_log --name $name --nclass 5 --nGPU 4 --bs 128 --ep 50000 --lr 0.001

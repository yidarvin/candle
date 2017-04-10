declare -r path_train=/home/dnr/Documents/data/eye_seg/training
#declare -r path_test=/home/dnr/Documents/data/eye_seg/testing
#declare -r path_val=/home/dnr/Documents/data/eye_seg/validation
#declare -r path_inf=/home/dnr/Documents/data/eye_seg/inference

declare -r name=eye-seg

declare -r path_model=/home/dnr/modelState/$name.ckpt
declare -r path_log=/home/dnr/logs/$name.txt

python /home/dnr/FirstAid/train_CNNsegmentation.py --pTrain $path_train --pModel $path_model --pLog $path_log --name $name --nGPU 4 --bs 128 --ep 500
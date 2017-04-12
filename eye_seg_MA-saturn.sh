declare -r path_train=/media/dnr/Documents/data/eye_seg_MA/training
#declare -r path_test=/home/dnr/Documents/data/eye_seg/testing
declare -r path_val=/media/dnr/Documents/data/eye_seg_MA/validation
#declare -r path_inf=/home/dnr/Documents/data/eye_seg/inference

declare -r name=MA-seg

declare -r path_model=/home/dnr/modelState/$name.ckpt
declare -r path_log=/home/dnr/logs/$name.txt
declare -r path_vis=/home/dnr/visualizations/$name

CUDA_VISIBLE_DEVICES=1 python /home/dnr/FirstAid/train_CNNsegmentation.py --pTrain $path_train --pVal $path_val --pModel $path_model --pLog $path_log --pVis $path_vis --name $name --nGPU 1 --bs 16 --ep 50000 --lr 0.001

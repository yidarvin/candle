declare -r path_train=/home/dnr/Documents/data/eye_seg_MA/training
#declare -r path_test=/home/dnr/Documents/data/eye_seg/testing
declare -r path_val=/home/dnr/Documents/data/eye_seg_MA/validation
#declare -r path_inf=/home/dnr/Documents/data/eye_seg_EX/inference
declare -r path_inf=/home/dnr/Documents/data/messidor/training

declare -r name=MA-seg

declare -r path_model=/home/dnr/modelState/$name.ckpt
declare -r path_log=/home/dnr/logs/$name.txt
declare -r path_vis=/home/dnr/visualizations/$name

#CUDA_VISIBLE_DEVICES=2,3 python /home/dnr/FirstAid/train_CNNsegmentation.py --pTrain $path_train --pVal $path_val --pModel $path_model --pLog $path_log --pVis $path_vis --name $name --nGPU 2 --bs 16 --ep 50000 --lr 0.001

CUDA_VISIBLE_DEVICES=2,3 python /home/dnr/FirstAid/train_CNNsegmentation.py --pInf $path_inf --pModel $path_model --pLog $path_log --pVis $path_vis --name $name --nGPU 2 --bs 16 --ep 50000 --lr 0.001

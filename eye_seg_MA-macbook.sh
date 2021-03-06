declare -r path_train=/Volumes/DATA000/data_processed/eye_seg_MA/training
#declare -r path_test=/home/dnr/Documents/data/eye_seg/testing
declare -r path_val=/Volumes/DATA000/data_processed/eye_seg_MA/validation
#declare -r path_inf=/home/dnr/Documents/data/eye_seg/inference

declare -r name=MA-seg

declare -r path_model=/home/dnr/modelState/$name.ckpt
declare -r path_log=/Users/yidarvin/Documents/logs/$name.txt
declare -r path_vis=/home/dnr/visualizations/$name

CUDA_VISIBLE_DEVICES=0,1,2,3 python /Users/yidarvin/Desktop/FirstAid/train_CNNsegmentation.py --pTrain $path_train --pVal $path_val --name $name --pLog $path_log --nGPU 0 --bs 4 --ep 50000 --lr 0.001

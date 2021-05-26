# Pulmonary nodule detection based on 3D Squeeze and Excitation Networks:

###1、 运行环境
Dependecies: Ubuntu 14.04, 
python 2.7, 
CUDA 8.0, cudnn 5.1, 
h5py (2.6.0), 
SimpleITK (0.10.0), 
numpy (1.11.3), 
nvidia-ml-py (7.352.0), 
matplotlib (2.0.0), scikit-image (0.12.3), 
scipy (0.18.1), 
pyparsing (2.1.4), 
pytorch (0.1.10+ac9245a) (anaconda is recommended)

###2、数据集来源

LUNA16 dataset from https://luna16.grand-challenge.org/data/

###3、代码运行
（1）数据预处理

 设置config_training.py 的路径
 python prepare.py
 
（2） 检测网络的训练
sh run_training.sh

其中 --model SE_inception/SE_res18/Inception

（3）模型评估
 python ./evaluationScript/frocwrtdetpepchluna16.py


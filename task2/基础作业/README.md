# 第二次作业：基础作业-基于自定义数据集 balloon 训练实例分割模型。
## 文件结构
mask_rcnn_r50_fpn_1x_coco.py 基础配置文件  
ballon.py                    训练使用的配置文件  
data.py                      将balloon数据集转为coco格式  
video_splash.py              给视频增加彩色气球特效  
20230208_221032.log.json     训练时精度日志  
[balloon_epoch_24.pth](https://pan.baidu.com/s/1STtSLS1GtyAd4oT73x1c5A) (提取码：s3m1) 最高精度模型权重   
[splash.mp4](https://pan.baidu.com/s/1STtSLS1GtyAd4oT73x1c5A) (提取码：s3m1) 增加特效后的模型 

## 模型配置
预训练模型：在MSCOCO数据集上预训练的maskRCNN  
训练环境：单卡V100，bathsize=12  
优化器：带动量的SGD算法，动量因子为0.9，学习率为0.001，权重衰退因子为0.0001，关闭梯度裁剪。  相较于原始模型的0.02学习率，由于是微调，大幅度降低学习率。  
学习策略：一共训练24个epoch，在第16和第22个epoch进行学习率衰减。

## 实验结果
在验证集上最高能够达到67.32%的bbox map和76.96%的segm map。  
可视化检测结果后发现，bbox精度之所以比segm低，是因为图中气球存在大量的重叠遮挡，这对bbox预测并不友好。

## 如何使用
### 1. 安装基础环境
创建conda环境
```shell
conda create -n mmdet python=3.8
conda activate mmdet
```
安装pytorch(请前往[官网](https://pytorch.org/get-started/previous-versions/)寻找符合自己cuda版本的pytorch.这里使用的torch1.0+cuda11.1)
```shell
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
安装openmim
```shell
pip install openmim
```
安装mmcv-full和mmdet
```shell
mim install mmcv-full
mim install mmdet
```
### 2. 处理数据集
点击下载[balloon数据集](https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip)
或者命令行下载
```shell
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
```
解压数据集
```shell
unzip balloon_dataset.zip
```
生成coco格式的annotation.(修改data.py中的原始json路径，训练集和验证集要分别转换，会在同样路径生成coco.json)
```shell
python data.py
```

### 3. 训练模型
修改ballon.py配置文件中训练集和测试集的路径为你本地实际路径。
下载预训练权重
```shell
mim download mmdet --config mask_rcnn_r50_fpn_1x_coco --dest ./checkpoint
```

修改ballon.py配置文件中load_from路径为下载好的预训练权重,然后训练模型
```shell
min train mmdet ballon.py
```

### 4. 视频特效
修改video_splash.py中的config路径，训练好的模型权重和视频路径。  
可以根据检测情况修改置信度阈值score_thr,阈值越高分割误检越少，但是容易漏检。
```python
config_file = "./ballon.py"
checkpoint_file = "./balloon_epoch_24.pth"
video_path  = "./test_video.mp4"
score_thr = 0.3
```
视频特效转换
```shell
python video_splash.py
```
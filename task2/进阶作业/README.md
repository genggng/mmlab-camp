# 第二次作业：进阶作业-在PASCAL VOC2012训练目标检测模型。
### 文件结构
retinanet_r50_fpn_1x_voc0712.py 基础配置文件  
voc2012.py                      训练配置文件  
20230209_115102.log.json        训练时精度日志  
[voc_2012_epoch_4.pth](https://pan.baidu.com/s/1STtSLS1GtyAd4oT73x1c5A) (提取码：s3m1) 最高精度模型权重    

### 模型配置
模型结构：使用retinaNet，backbone使用经过ImageNet-1K预训练的ResNet50，检测头未经过预训练 
训练环境：单卡V100，bathsize=32  
优化器：带动量的SGD算法，动量因子为0.9，学习率为0.01，权重衰退因子为0.0001，关闭梯度裁剪  
学习策略：一共训练4个epoch，在第3个epoch进行学习率衰减。由于使用RepeatDataset，并且设置重复次数为3，相当于12个epoch。

### 实验结果  
在Pascal VOC的train集进行训练，在val集上最高能够达到36.96%的mAP。
# 第一次作业：进阶作业-使用 MMClassification 训练cifar=10分类模型。
### 文件结构
resnet50_8xb16-mixup_cifar10-new.py 配置文件  
20230206_161841.log 终端输出日志  
20230206_161841.log.json 训练时精度日志  
[cifra-epoch-171.pth](https://pan.baidu.com/s/1STtSLS1GtyAd4oT73x1c5A) (提取码：s3m1) 最高精度模型权重  
 最高精度模型权重  

### 模型配置
数据增强：训练时使用随机裁切和随机翻转
模型结构：未经过预训练的resnet50，同时使用batchmixup策略  
训练环境：单卡2080ti，bathsize=128  
优化器：带动量的SGD算法，动量因子为0.9，学习率为0.001，权重衰退因子为0.0001，关闭梯度裁剪  
学习策略：一共训练200个epoch，在第100和第150个epoch进行学习率衰减。

### 实验结果  
在测试集上最高top-1准确度能够达到96.39%。
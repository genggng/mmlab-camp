# 第一次作业：基础作业-使用 MMClassification 训练花卉图片分类模型。
### 文件结构
resnet18_b32_flower.py 配置文件  
20230205_191002.log 终端输出日志  
20230205_191002.log.json 训练时精度日志  
[flower-epoch-8.pth](https://pan.baidu.com/s/1STtSLS1GtyAd4oT73x1c5A) (提取码：s3m1) 最高精度模型权重  

### 模型配置
预训练模型：在imagenet-1K上8卡32batchszie训练的resnet18  
训练环境：单卡2080ti，bathsize=32  
优化器：带动量的SGD算法，动量因子为0.9，学习率为0.001，权重衰退因子为0.0001，关闭梯度裁剪  
学习策略：一共训练12个epoch，在第8和第10个epoch进行学习率衰减。

### 实验结果
flower 数据集包含 5 种类别的花卉图像：雏菊 daisy 588张，蒲公英 dandelion 556张，玫瑰 rose 583张，向日葵 sunflower 536张，郁金香 tulip 585张。  
将每个类别随机按照8:2的比例随机划分为训练集和验证集，在训练集进行训练。  
在测试集上最高top-1准确度能够达到95.1%。
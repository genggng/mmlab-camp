# 第三次作业：进阶作业-使用MMSegmentation，在PascalVOC2012数据集上，训练语义分割模型。
### 文件结构
mmseg_voc12.py                   训练配置文件    
20230212_215851.log             终端日志文件   
20230212_215851.json            训练时精度日志    
[iter_20000.pth](https://pan.baidu.com/s/1STtSLS1GtyAd4oT73x1c5A) (提取码：s3m1) 最高精度模型权重    


### 数据与模型配置
数据集：PascalVOC2012数据集 
模型结构：使用pspnet，backbone使用经过ImageNet-1K预训练的ResNet50，检测头未经过预训练 。  
训练环境：8卡V100，单卡bathsize=20，总batchsize=160。   
优化器：带动量的SGD算法，动量因子为0.9，学习率为0.01，权重衰退因子为0.0005
学习策略：一共训练20000个iter，每2000个iter做一次验证。

### 实验结果  
在PascalVOC2012验证集集上最高能够达到90.79%的aAcc,59.5%的mIoU和67.95%mAcc。  
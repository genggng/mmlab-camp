# 第三次作业：基础作业-使用MMSegmentation，在自己的数据集上，训练语义分割模型。
### 文件结构
new_cfg.py.py                    训练配置文件    
20230212_203802.log              终端日志文件   
20230209_115102.log.json        训练时精度日志   
fps_20230212_203525.json        模型速度测试日志   
[iter_1600.pth](https://pan.baidu.com/s/1STtSLS1GtyAd4oT73x1c5A) (提取码：s3m1) 最高精度模型权重    


### 数据与模型配置
数据集：组织病理切片小鼠肾小球数据集，按照8:2随机划分为训练集和验证集。  
模型结构：使用pspnet，backbone使用经过ImageNet-1K预训练的ResNet50，检测头未经过预训练 。  
训练环境：单卡1070ti，bathsize=8。   
优化器：带动量的SGD算法，动量因子为0.9，学习率为0.01，权重衰退因子为0.0005
学习策略：一共训练1600个iter，每400个iter做一次验证。

### 实验结果  
在组织病理切片小鼠肾小球数据集进行训练，在验证集集上最高能够达到99.45%的aAcc,84.51%的mIoU和88.39%mAcc。  
在1070ti上能够达到每秒27帧的推理速度。
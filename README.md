# mmlab-camp
mmalb 实战营作业代码,包含三次作业的代码。
# task1：图像分类
使用 MMClassification 训练花卉图片分类模型。
请参考 MMClassification 的文档以及教学视频，整理训练数据集，修改配置文件，基于 MMClassification 提供的预训练模型，在 flowers 数据集上完成分类模型的微调训练。

同学们可以基于北京超算平台完成任务，本地有显卡的同学也可以在本地环境完成作业。超算平台的使用手册在这里可以找到：https://aicarrier.feishu.cn/docs/doccnP7NPMfRr9TAcwRsPKgkOgc?from=from_copylink

## 1. 整理 flower 数据集

#### 数据集介绍

flower 数据集包含 5 种类别的花卉图像：雏菊 daisy 588张，蒲公英 dandelion 556张，玫瑰 rose 583张，向日葵 sunflower 536张，郁金香 tulip 585张。

数据集下载链接：

- 国际网：https://www.dropbox.com/s/snom6v4zfky0flx/flower_dataset.zip?dl=0
- 国内网：https://pan.baidu.com/s/1RJmAoxCD_aNPyTRX6w97xQ 提取码: 9x5u

#### 对数据集进行划分

1. 将数据集按照 8:2 的比例划分成训练和验证子数据集，并将数据集整理成 ImageNet的格式。这个过程同学们可以通过 Python 或其他脚本程序完成。具体步骤如下：

2. 将训练子集和验证子集放到 train 和 val 文件夹下。

文件结构如下：

```
flower_dataset
|--- classes.txt
|--- train.txt
|--- val.txt
|    |--- train
|    |    |--- daisy
|    |    |    |--- NAME1.jpg
|    |    |    |--- NAME2.jpg
|    |    |    |--- ...
|    |    |--- dandelion
|    |    |    |--- NAME1.jpg
|    |    |    |--- NAME2.jpg
|    |    |    |--- ...
|    |    |--- rose
|    |    |    |--- NAME1.jpg
|    |    |    |--- NAME2.jpg
|    |    |    |--- ...
|    |    |--- sunflower
|    |    |    |--- NAME1.jpg
|    |    |    |--- NAME2.jpg
|    |    |    |--- ...
|    |    |--- tulip
|    |    |    |--- NAME1.jpg
|    |    |    |--- NAME2.jpg
|    |    |    |--- ...
|    |--- val
|    |    |--- daisy
|    |    |    |--- NAME1.jpg
|    |    |    |--- NAME2.jpg
|    |    |    |--- ...
|    |    |--- dandelion
|    |    |    |--- NAME1.jpg
|    |    |    |--- NAME2.jpg
|    |    |    |--- ...
|    |    |--- rose
|    |    |    |--- NAME1.jpg
|    |    |    |--- NAME2.jpg
|    |    |    |--- ...
|    |    |--- sunflower
|    |    |    |--- NAME1.jpg
|    |    |    |--- NAME2.jpg
|    |    |    |--- ...
|    |    |--- tulip
|    |    |    |--- NAME1.jpg
|    |    |    |--- NAME2.jpg
|    |    |    |--- ...
```

3. 创建并编辑标注文件将所有类别的名称写到 `classes.txt` 中，每行代表一个类别。

4. 生成训练（可选）和验证子集标注列表 `train.txt` 和  `val.txt` ，每行应包含一个文件名和其对应的标签。样例：

```
...
daisy/NAME**.jpg 0
daisy/NAME**.jpg 0
...
dandelion/NAME**.jpg 1
dandelion/NAME**.jpg 1
...
rose/NAME**.jpg 2
rose/NAME**.jpg 2
...
sunflower/NAME**.jpg 3
sunflower/NAME**.jpg 3
...
tulip/NAME**.jpg 4
tulip/NAME**.jpg 4
```

为了节约线上时间，这个过程可以在本地完成。

整理完成后，将处理好的数据集迁移到 `mmclassification/data ` 文件夹下。

## 2. 构建模型微调的配置文件

使用 `_base_` 继承机制构建用于微调的配置文件，可以继承任何 MMClassification 提供的基于 ImageNet 的配置文件并进行修改。

### 修改模型配置

修改分类头，将模型适应为 flowers 中的数据类别数。

### 修改数据配置

修改训练和验证集的数据路径，数据集标注列表，以及类别名文件路径。

将评估方法修改为仅使用 top-1 分类错误率。

### 学习率策略

微调一般会使用更小的学习率和更少的训练周期。因此请在配置文件中修改学习率和训练周期。

### 配置预训练模型

从 Model Zoo 中找到原配置文件对应的模型文件，并下载到平台或本地环境中，通常放置在 `checkpoints` 文件夹中。

在配置文件中配置预训练模型的路径，完成 finetune 训练。


## 3. 使用工具进行模型微调

使用 `tools/train.py` 进行模型微调，并通过 `work_dir` 参数指定工作路径，该路径中会保存训练的模型。调整参数，或使用不同的预训练模型，尝试获得更高的分类精度。

作为参考，在该数据集上达到 90% 以上的分类精度并不困难。

## 4. 作业提交和评价标准

完成模型微调之后，请同学们将配置文件、训练保存的模型和log文件，全部打包提交到自己的 github （如没有，需要新建一个），而后提交至 issue 中对应作业和班级的位置。

# task2：目标检测

请参考 MMDetection 文档及教程，基于自定义数据集 balloon 训练实例分割模型，基于训练的模型在样例视频上完成color splash的效果制作，即使用模型对图像进行逐帧实例分割，并将气球以外的图像转换为灰度图像。

color splash样例：
https://github.com/matterport/Mask_RCNN/blob/master/assets/balloon_color_splash.gif 


**注：由于GPU使用资源有限，请同学们尽量在CPU上先调通程序再进行模型的训练。**

## 1. Balloon数据集

balloon是带有mask的气球数据集，其中训练集包含61张图片，验证集包含13张图片。

下载链接：https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip 

## 2. 新数据集的支持

由于需要支持新的数据集，请同学们选择MMDetection支持的以下方法来操作:

1. 将数据集整理为COCO格式
2. 将数据集整理为中间格式
3. 直接实现新数据集的支持
注：方法2，3不直接支持分割任务，需要后处理。这里推荐使用方法1。

## 3. 构建配置文件

从 Model Zoo 中找到 mask rcnn 模型并找到 `configs/mask_rcnn/` 中对应的模型配置文件。将模型下载到环境中，通常放在放置在 `checkpoints` 文件夹中。

可以参考：https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn 


构建配置文件可以使用继承机制，从选择模型配置中继承，并修改自定义数据集需要的内容。



## 4. 模型微调

通过命令行工具或者 Python API 完成模型微调，并在验证集上完成测试和评价。可以调整参数，或使用不同的预训练模型来获得更高的评分。


## 5. Color splash特效制作

在获取到图像的mask之后，请同学们将图像转为灰度图像，并在mask区域将原图像的像素值拷贝到灰度图像上即可完成特效制作。


测试样例视频请同学们在[作业二](https://github.com/open-mmlab/OpenMMLabCamp/tree/main/AI%20%E5%AE%9E%E6%88%98%E8%90%A5%E5%9F%BA%E7%A1%80%E7%8F%AD/%E4%BD%9C%E4%B8%9A%E4%BA%8C%E8%B5%84%E6%96%99)文件夹中下载 `test_video.mp4`


有关视频的读写可以参考以下文档：

- mmcv: 基于opencv的实现 

https://github.com/open-mmlab/mmcv/blob/master/mmcv/video/io.py 

- opencv:

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html# 

- skvideo: 需要额外安装ffmepg，且需要注意设置压制参数 `pix_fmt` 为 `yuv420p` 以支持主流播放器

http://www.scikit-video.org/stable/modules/io.html#module-skvideo.io 


## 6. 作业提交：

完成视频特效制作之后，请同学们将以下文件上传至自己的github，并将链接按要求提交在对应班级issue：

* 训练模型的配置文件
* 训练好的模型文件（若文件过大，可以存至网盘，将链接放在github readme中）
* 特效制作后的视频文件（若文件过大，可以存至网盘，将链接放在github readme中）
* log文件

# task3: 语义分割
使用MMSegmentation，在自己的数据集上，训练语义分割模型
1. 数据集标注（可选）

使用Labelme、LabelU等数据标注工具，标注多类别语义分割数据集，并保存为指定的格式。

2. 数据集整理

划分训练集、测试集

3. 使用MMSegmentation训练语义分割模型

在MMSegmentation中，指定预训练模型，配置config文件，修改类别数、学习率。

4. 用训练得到的模型预测

获得测试集图片或新图片的语义分割预测结果，对结果进行可视化和后处理。

5. 在测试集上评估算法的速度和精度性能

6. 使用MMDeploy部署语义分割模型（可选）



本课代码：https://github.com/TommyZihao/MMSegmentation_Tutorials/tree/main/20230206

# 参考单类别语义分割数据集
组织病理切片小鼠肾小球：https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/dataset/Glomeruli-dataset.zip

电子显微镜粒子：https://www.kaggle.com/datasets/batuhanyil/electron-microscopy-particle-segmentation

农作物病虫害叶片：https://www.kaggle.com/datasets/fakhrealam9537/leaf-disease-segmentation-dataset

农作物地块：https://www.kaggle.com/datasets/khlaifiabilel/pastis

水下场景：https://www.kaggle.com/datasets/ashish2001/semantic-segmentation-of-underwater-imagery-suim

西红柿种子：https://www.kaggle.com/datasets/juanma9901/tomatoseedsdatasetjm

肾小球：https://www.kaggle.com/datasets/baesiann/glomeruli-hubmap-external-1024x1024

卫星建筑物：https://www.kaggle.com/datasets/hyyyrwang/buildings-dataset

荧光显微镜小鼠脑切片发光神经元-实例分割：https://www.kaggle.com/datasets/nbroad/fluorescent-neuronal-cells

混凝土裂缝：https://www.kaggle.com/datasets/jakubniemiec/concrete-crack-images

# 参考多类别语义分割数据集
高分辨率航拍-多类别：https://www.kaggle.com/datasets/titan15555/uavid-semantic-segmentation-dataset

衣物：https://www.kaggle.com/datasets/rajkumarl/people-clothing-segmentation

海洋生物：https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset

腿和脚趾：https://www.kaggle.com/datasets/tapakah68/legs-segmentation

无人机航拍：https://www.kaggle.com/datasets/santurini/semantic-segmentation-drone-dataset

无人机航拍：https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset

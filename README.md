# SPL-ESP-BC
为了避免难样本对飞鸟目标检测模型训练过程的影响（飞鸟目标检测模型[FBOD](https://github.com/Ziwei89/FBOD)，在前期工作中，我们根据监控视频中飞鸟目标的特点设计了FBOD模型），提出了一种基于置信度的简单样本先验自步学习方法(SPL-ESP-BC)，该方法是一种新的模型训练策略。首先，改进了自步学习(Self-Paced Learning, SPL)中基于损失的最小化函数，提出了基于置信度的最小化函数，使其更适合于单类目标检测任务。其次，提出了一种简单样本先验(Easy Sample Prior, ESP)的SPL策略，首先采用普通的训练方法，使用简单样本对FBOD模型进行训练，然后采用SPL的模型训练方法，使用所有样本对其进行训练，使模型在采用SPL训练方法的初期，具有判断简单样本和难样本的能力。最后，将ESP策略与基于置信度的最小化函数相结合，提出了SPL-ESP-BC模型训练策略，利用该策略对FBOD模型进行训练，可以使其从易到难更好地学习监控视频中飞鸟目标的特征。  


该项目是在项目[FBOD](https://github.com/Ziwei89/FBOD)基础上，进一步设计的新的模型训练策略。大多数参数含义和设置与[FBOD](https://github.com/Ziwei89/FBOD)相同，该项目将不再赘述。模型训练时，与本项目无关的参数（与模型训练策略无关的参数)采用默认参数。若想了解这些参数请访问：https://github.com/Ziwei89/FBOD  

另外本项目默认采用了Frames padding方法。  

# 项目应用步骤

## 1、克隆项目到本地
```
git clone https://github.com/Ziwei89/FBOD-BSPL.git
```
## 2、准备训练和测试数据

同项目[FBOD](https://github.com/Ziwei89/FBOD)。

## 3、不同训练策略训练模型
对比四种模型训练策略，分别是：所有样本普通训练策略，简单样本训练策略，基于置信度的自步学习策略，困难样本挖掘训练策略以及普通自步学习策略。  
所有样本普通训练策略，简单样本训练策略，基于置信度的自步学习策略三种模型训练策略使用脚本TrainFramework/train_AP50.py；困难样本挖掘训练策略，普通自步学习策略两种模型训练策略使用脚本TrainFramework/train_AP50_HEM_SPL.py  

**<font color=red>注意：</font>基于置信度的简单样本先验自步学习策略(SPL-ESP-BC)，有两个训练策略组成，先采用简单样本训练策略训练模型，然后采用基于置信度的自步学习策略训练模型(简单样本训练策略训练的模型作为预训练模型)。**

相关参数解释如下(在设置时请参考TrainFramework/config/opts.py文件)：  
```
pretrain_model_path                #预训练模型的路径。在置信度的简单样本先验自步学习策略，需使用简单样本训练策略训练的模型
Add_name                           #在相关记录文件(如模型保存文件夹或训练记录图片)，增加后缀
learn_mode                         #模型学习策略：
                                            All_Sample：所有样本普通训练策略
                                            Easy_sample：简单样本训练策略
                                            SPLBC：基于置信度自步学习训练策略
                                            SPL：普通自步学习策略
                                            HEM：困难样本挖掘模型训练策略
spl_mode                            #自步学习正则化器，普通自步学习策略时有效: hard, linear, logarithmic
```
三个训练的例子：  
* 简单样本训练策略： 
```
cd TrainFramework
python train_AP50.py \
        --learn_mode=Easy_sample \
        --Add_name=20240102
cd .
```

* 简单样本训练策略： 
```
cd TrainFramework
python train_AP50.py \
        --learn_mode=SPLBC \
        --pretrain_model_path=./logs/five/384_672/RGB_relatedatten_cspdarknet53_concat_Easy_Sample_aa_20240102/FB_object_detect_model.pth \
        --Add_name=20240104
cd .

* 普通自步学习策略 
```
cd TrainFramework
python train_AP50.py \
        --learn_mode=SPL \
        --spl_mode=hard \
        --Add_name=20240104
cd .
```

## 未完待续...
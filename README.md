# SPL-ESP-BC
为了避免难样本对飞鸟目标检测模型训练过程的影响（飞鸟目标检测模型[FBOD](https://github.com/Ziwei89/FBOD)，在前期工作中，我们根据监控视频中飞鸟目标的特点设计了FBOD模型），提出了一种基于置信度的简单样本先验自步学习方法(SPL-ESP-BC)，该方法是一种新的模型训练策略。首先，改进了自步学习(Self-Paced Learning, SPL)中基于损失的最小化函数，提出了基于置信度的最小化函数，使其更适合于单类目标检测任务。其次，提出了一种简单样本先验(Easy Sample Prior, ESP)的SPL策略，首先采用普通的训练方法，使用简单样本对FBOD模型进行训练，然后采用SPL的模型训练方法，使用所有样本对其进行训练，使模型在采用SPL训练方法的初期，具有判断简单样本和难样本的能力。最后，将ESP策略与基于置信度的最小化函数相结合，提出了SPL-ESP-BC模型训练策略，利用该策略对FBOD模型进行训练，可以使其从易到难更好地学习监控视频中飞鸟目标的特征。  


该项目是在项目[FBOD](https://github.com/Ziwei89/FBOD)基础上，进一步设计的新的模型训练策略。大多数参数含义和设置与[FBOD](https://github.com/Ziwei89/FBOD)，该项目将不再赘述，模型训练是采用默认参数。若想了解这些参数请访问：https://github.com/Ziwei89/FBOD  

另外本项目默认采用了Frames padding方法。  

# 项目应用步骤

## 1、克隆项目到本地
```
git clone https://github.com/Ziwei89/FBOD-BSPL.git
```
## 2、准备训练和测试数据

同项目[FBOD](https://github.com/Ziwei89/FBOD)。

## 未完待续...
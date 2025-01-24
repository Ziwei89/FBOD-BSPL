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
**<font color=red>注意：</font> 后续运行脚本前，初始工作位置默认在项目根目录(FBOD/)**

可以使用labelImg对图片进行标注，得到xml文件。  
标签文件中应包含目标边界框、目标类别以及目标的难度信息(在以后的项目中会用到目标的难度信息，在本项目中标注时不用考虑，相关代码会设置一个用不到的默认值)

### (1) 数据组织
```  
data_root_path/  
               videos/
                     train/
                           bird_1.mp4
                           bird_2.mp4
                           ...  
                     val/
                images/
                     train/  
                           bird_1_000000.jpg  
                           bird_1_000001.jpg  
                           ...  
                           bird_2_000000.jpg  
                           ...  
                     val/
                labels
                     train/  
                           bird_1_000000.xml  
                           bird_1_000001.xml  
                           ...
                           bird_2_000000.xml
                           ...  
                     val/
```  
### (2) 生成数据描述txt文件用于训练和测试(训练一个txt,测试一个txt)
数据描述txt文件的格式如下：  
连续n帧图像的第一帧图像名字 *空格* 中间帧飞鸟目标信息  
image_name x1,y1,x2,y2,cls,difficulty x1,y1,x2,y2,cls,difficulty  
eg:  
```
...  
bird_3_000143.jpg 995,393,1016,454,0,0.625
bird_3_000144.jpg 481,372,489,389,0,0.375 993,390,1013,456,0,0.625
...  
bird_40_000097.jpg None
...
```
我们提供了一个脚本，可以生成这样的数据描述txt。该脚本为Data_process目录下的continuous_image_annotation.py (脚本continuous_image_annotation_frames_padding.py增加了序列padding, 序列padding就是在视频的开头前和结尾后增加一些全黑的图片，使前几帧和后几帧有输出结果，具体请参考我们的论文)，运行该脚本需要指定数据集路径以及模型一次推理所输入的连续图像帧数：  
```
cd Data_process #从项目根目录进入数据处理目录
python continuous_image_annotation.py \
       --data_root_path=../dataset/FBD-SV-2024/ \
       --input_img_num=5
```
运行该脚本后，将在TrainFramework/dataloader/目录下生成两个txt文件，分别是img_label_five_continuous_difficulty_train_raw.txt和img_label_five_continuous_difficulty_val_raw.txt文件。这两个文件中的训练样本排列是顺序的，最好通过运行以下脚本将其打乱：  
```
cd TrainFramework/dataloader/ #从项目根目录进入训练框架下dataloader目录
python shuffle_txt_lines.py \
       --input_img_num=5
```
运行该脚本后，将在TrainFramework/dataloader/目录下生成img_label_five_continuous_difficulty_train.txt和img_label_five_continuous_difficulty_val.txt两个文件。
### (3) 准备类别txt文件
在TrainFramework/目录下创建model_data文件夹，然后再在TrainFramework/model_data/目录下创建名为classes.txt的文件，该文件中记录类别,如：
```
bird
```

## 3、不同训练策略训练模型
对比四种模型训练策略，分别是：所有样本普通训练策略，简单样本训练策略，基于置信度的自步学习策略，困难样本挖掘训练策略以及普通自步学习策略。  
所有样本普通训练策略，简单样本训练策略，基于置信度的自步学习策略三种模型训练策略使用脚本TrainFramework/train_AP50.py；困难样本挖掘训练策略，普通自步学习策略两种模型训练策略使用脚本TrainFramework/train_AP50_HEM_SPL.py  

**<font color=red>注意：</font>基于置信度的简单样本先验自步学习策略(SPL-ESP-BC)，有两个训练策略组成，先采用简单样本训练策略训练模型，然后采用基于置信度的自步学习策略训练模型(简单样本训练策略训练的模型作为预训练模型)。**

相关参数解释如下(在设置时请参考TrainFramework/config/opts.py文件)：  
```
data_root_path                     #数据集根路径
pretrain_model_path                #预训练模型的路径。在置信度的简单样本先验自步学习策略，需使用简单样本训练策略训练的模型
Add_name                           #在相关记录文件(如模型保存文件夹或训练记录图片)，增加后缀
learn_mode                         #模型学习策略：
                                            All_Sample：所有样本普通训练策略
                                            Easy_Sample：简单样本训练策略
                                            SPLBC：基于置信度自步学习训练策略
                                            SPL：普通自步学习策略
                                            HEM：困难样本挖掘模型训练策略
spl_mode                            #自步学习正则化器，普通自步学习策略时有效: hard, linear, logarithmic
```
另外两个参数m和r,是关于最小化函数和训练调度函数的。为了保持于论文表述一致，请保持使用默认参数。

三个训练的例子：  
* 简单样本训练策略： 
```
cd TrainFramework
python train_AP50.py \
        --data_augmentation=True \
        --data_root_path=../dataset/FBD-SV-2024/ \
        --learn_mode=Easy_Sample \
        --Add_name=20240102
cd ../
```

* 基于置信度的自步学习训练策略： 
```
cd TrainFramework
python train_AP50.py \
        --data_augmentation=True \
        --data_root_path=../dataset/FBD-SV-2024/ \
        --learn_mode=SPLBC \
        --pretrain_model_path=./logs/five/384_672/RGB_relatedatten_cspdarknet53_concat_Easy_Sample_aa_20240102/FB_object_detect_model.pth \
        --Add_name=20240104
cd ../
```
* 普通自步学习策略 
```
cd TrainFramework
python train_AP50_HEM_SPL.py \
        --data_augmentation=True \
        --data_root_path=../dataset/FBD-SV-2024/ \
        --learn_mode=SPL \
        --spl_mode=hard \
        --Add_name=20240104
cd ../
```
## 4、测试模型检测性能（测试模型时，参数设置要和对应模型训练时的参数一致）
```
cd TrainFramework
python mAP_for_AllVideo_coco_tools.py \
        --data_root_path=../dataset/FBD-SV-2024/ \
        --learn_mode=SPLBC \
        --Add_name=20240104 \
        --model_name=FB_object_detect_model.pth
cd ../
```
#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os
from config.opts import opts
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from net.FBODInferenceNet import FBODInferenceBody
from utils.FBODLoss import LossFunc
from FB_detector import FB_Postprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import FBObj
from dataloader.dataset_bbox import CustomDataset, dataset_collate
from utils.common import load_data_raw_resize_boxes, GetMiddleImg_ModelInput_for_MatImageList
from utils.get_box_info_list import getBoxInfoListForOneImage_Loss, image_info
from mAP import mean_average_precision
import copy
import math
import shutil

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# def adjust_spl_threshold(step_proportion=0.01):
#     """
#     spl_threshold
#         |_____________1
#         |           /:
#         |         /  :
#         |       /    :
#         |     /      :
#         |   /        :
#     0.2 |_/          :
#         |_:__________:______ Step_proportion
#          0.1         0.9
#     """
#     if step_proportion <= 0.1:
#         return 0.2
#     elif step_proportion <= 0.9:
#         return step_proportion + 0.1
#     else:
#         return 1.

### The above function adjust_spl_threshold is a special case when the argument lambda0=0.2, e1=0.1, e2=0.9 and r=1.
def adjust_spl_threshold(lambda0=0.2, e1=0.1, e2=0.9, step_proportion=0.01, r=1):
    """
    spl_threshold, spl based on loss

    """
    if e1 < 0:
        raise("Error! e1 must be larger than or equal to 0!")
    if e2 <= e1:
        raise("Error! e2 must be large e1!")
    if step_proportion <= e1:
        return lambda0
    elif step_proportion <= e2:
        return (1-(1-lambda0)*((e2-step_proportion)/(e2-e1))**r)
    else:
        return 1.


def spl_sampleWeight_Hard(sample_loss, spl_threshold):
    if sample_loss < spl_threshold:
        return 1
    else:
        return 0

def spl_sampleWeight_Linear(sample_loss, spl_threshold):
    if sample_loss < spl_threshold:
        return (1 - sample_loss/spl_threshold)
    else:
        return 0

def spl_sampleWeight_Continuous(sample_loss, spl_threshold, b_parameter=0.5):
    if b_parameter>=1 or b_parameter<=0:
        raise("b_parameter must be in range (0, 1)!")
    if sample_loss < spl_threshold*(1-b_parameter):
        return 1
    elif sample_loss >= spl_threshold*(1-b_parameter) and sample_loss <= spl_threshold:
        return (1/b_parameter)*(1 - sample_loss/spl_threshold)
    else:
        return 0

def spl_sampleWeight_Logarithmic(sample_loss, spl_threshold):
    if sample_loss < spl_threshold:
        parameter2 = 1- spl_threshold
        weight = (math.log(sample_loss + parameter2)/math.log(parameter2+0.01))
        return weight
    else:
        return 0

def spl_sampleWeight_Polynomial(sample_loss, spl_threshold, t_parameter=3):
    if t_parameter<=1:
        raise("t_parameter must be larger than 1!")
    if sample_loss < spl_threshold:
        weight = (1-sample_loss/spl_threshold)**(1/(t_parameter-1))
        return weight
    else:
        return 0

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class LablesToResults(object):
    def __init__(self, batch_size):#h,w
        self.batch_size = batch_size

    def covert(self, labels_list, iteration): # TO Raw image size
        label_obj_list = []
        for batch_id in range(self.batch_size):
            labels = labels_list[batch_id]
            if labels.size==0:
                continue
            image_id = self.batch_size*iteration + batch_id
            for label in labels:
                # class_id = label[4] + 1 ###Include background in this project, the label didn't include background classes.
                box = [label[i] for i in range(4)]
                label_obj_list.append(FBObj(score=1., image_id=image_id, bbox=box))
        return label_obj_list

def fit_one_epoch(largest_AP_50,net,loss_func_train,loss_func_val,epoch,epoch_size,epoch_size_val,gen,genval,
                  Epoch,cuda,save_model_dir,labels_to_results,detect_post_process):
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets, names = batch[0], batch[1], batch[2]
            #print(images.shape) 1,7,384,672
            with torch.no_grad():
                # if cuda:
                #     images = Variable(images).to(torch.device('cuda:0'))
                #     targets = [Variable(fature_label.to(torch.device('cuda:0'))) for fature_label in targets] ## 
                # else:
                #     images = Variable(images)
                #     targets = [Variable(fature_label)for fature_label in targets] ##
                if cuda:
                    images = Variable(torch.from_numpy(images)).to(torch.device('cuda:0'))
                    targets = [Variable(torch.from_numpy(fature_label)) for fature_label in targets] ## 
                else:
                    images = Variable(torch.from_numpy(images))
                    targets = [Variable(torch.from_numpy(fature_label).type(torch.FloatTensor)) for fature_label in targets] ##
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_func_train(outputs, targets)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_loss += loss
            waste_time = time.time() - start_time
            
            pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1), 
                                'lr'        : get_lr(optimizer),
                                'step/s'    : waste_time})
            pbar.update(1)

            start_time = time.time()
    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        all_label_obj_list = []
        all_obj_result_list = []
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]
            labels_list = copy.deepcopy(targets_val)
            with torch.no_grad():
                # if cuda:
                #     images_val = Variable(images_val).to(torch.device('cuda:0'))
                #     targets_val = [Variable(fature_label.to(torch.device('cuda:0'))) for fature_label in targets_val] ## 
                # else:
                #     images_val = Variable(images_val)
                #     targets_val = [Variable(fature_label)for fature_label in targets_val] ##
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val)).to(torch.device('cuda:0'))
                    targets_val = [Variable(torch.from_numpy(fature_label)) for fature_label in targets_val] ## 
                else:
                    images_val = Variable(torch.from_numpy(images_val))
                    targets_val = [Variable(torch.from_numpy(fature_label).type(torch.FloatTensor)) for fature_label in targets_val] ##
                optimizer.zero_grad()
                outputs = net(images_val)
                loss = loss_func_val(outputs, targets_val)
                val_loss += loss

                if (epoch+1) >= 30:
                    label_obj_list = labels_to_results.covert(labels_list, iteration)
                    all_label_obj_list += label_obj_list

                    obj_result_list = detect_post_process.Process(outputs, iteration)
                    all_obj_result_list += obj_result_list

            pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)
    net.train()
    if (epoch+1) >= 30:
        AP_50,REC_50,PRE_50=mean_average_precision(all_obj_result_list,all_label_obj_list,iou_threshold=0.5)
    else:
        AP_50,REC_50,PRE_50 = 0,0,0
    
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f  || AP_50: %.4f  || REC_50: %.4f  || PRE_50: %.4f' % (total_loss/(epoch_size+1), val_loss/(epoch_size_val+1),  AP_50, REC_50, PRE_50))
    
    if (epoch+1)%10 == 0 or epoch == 0:
        if largest_AP_50 < AP_50:
            largest_AP_50 = AP_50
        print('Saving state, iter:', str(epoch+1))
        torch.save(model.state_dict(), save_model_dir + 'Epoch%d-Total_Loss%.4f-Val_Loss%.4f-AP_50_%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1),AP_50))
        torch.save(model.state_dict(), save_model_dir + 'FB_object_detect_model.pth')
    else:
        if largest_AP_50 < AP_50:
            largest_AP_50 = AP_50
            print('Saving state, iter:', str(epoch+1))
            torch.save(model.state_dict(), save_model_dir + 'Epoch%d-Total_Loss%.4f-Val_Loss%.4f-AP_50_%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1),AP_50))
            torch.save(model.state_dict(), save_model_dir + 'FB_object_detect_model.pth')
    if (epoch+1) >= 30:
        return total_loss/(epoch_size+1), val_loss/(epoch_size_val+1), largest_AP_50, AP_50
    else:
        return total_loss/(epoch_size+1), val_loss/(epoch_size_val+1), largest_AP_50, 0.80

num_to_english_c_dic = {3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}

####################### Plot figure #######################################
x_epoch = []
record_loss = {'train_loss':[], 'test_loss':[]}
fig = plt.figure()

ax0 = fig.add_subplot(111, title="Train the FB_object_detect model")
ax0.set_ylabel('loss')
ax0.set_xlabel('Epochs')

def draw_curve_loss(epoch, train_loss, test_loss, pic_name):
    global record_loss
    record_loss['train_loss'].append(train_loss)
    record_loss['test_loss'].append(test_loss)

    x_epoch.append(int(epoch))
    ax0.plot(x_epoch, record_loss['train_loss'], 'b', label='train')
    ax0.plot(x_epoch, record_loss['test_loss'], 'r', label='val')
    if epoch == 1:
        ax0.legend()
    fig.savefig(pic_name)
########============================================================########
x_ap50_epoch = []
record_ap50 = {'AP_50':[]}
fig_ap50 = plt.figure()

ax1 = fig_ap50.add_subplot(111, title="Train the FB_object_detect model")
ax1.set_ylabel('ap_50')
ax1.set_xlabel('Epochs')

def draw_curve_ap50(epoch, ap_50, pic_name):
    global record_ap50
    record_ap50['AP_50'].append(ap_50)

    x_ap50_epoch.append(int(epoch))
    ax1.plot(x_ap50_epoch, record_ap50['AP_50'], 'g', label='AP_50')
    if epoch == 30:
        ax1.legend()
    fig_ap50.savefig(pic_name)
#############################################################################

if __name__ == "__main__":

    opt = opts().parse()
    # assign_method: The label assign method. binary_assign, guassian_assign or auto_assign
    if opt.assign_method == "auto_assign":
        abbr_assign_method = "aa"
    else:
        raise("Error! assign_method error.")
    if opt.learn_mode == "SPL":
        Add_name = opt.spl_mode + "_" + opt.Add_name
    else:
        Add_name = opt.Add_name
    
    save_model_dir = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                             + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + opt.learn_mode + "_" + abbr_assign_method + "_"  + Add_name + "/"
    os.makedirs(save_model_dir, exist_ok=True)

    ############### For log figure ################
    log_pic_name_loss = save_model_dir + "loss.jpg"
    log_pic_name_ap50 = save_model_dir + "ap50.jpg"
    ################################################

    config_txt = save_model_dir + "config.txt"
    if os.path.exists(config_txt):
        pass
    else:
        config_txt_file = open(config_txt, 'w')
        config_txt_file.write("Input mode: " + opt.input_mode + "\n")
        config_txt_file.write("Data root path: " + opt.data_root_path + "\n")
        config_txt_file.write("Aggregation method: " + opt.aggregation_method + "\n")
        config_txt_file.write("Backbone name: " + opt.backbone_name + "\n")
        config_txt_file.write("Fusion method: " + opt.fusion_method + "\n")
        config_txt_file.write("Assign method: " + opt.assign_method + "\n")
        config_txt_file.write("Scale factor: " + str(opt.scale_factor) + "\n")
        config_txt_file.write("Batch size: " + str(opt.Batch_size) + "\n")
        config_txt_file.write("Data augmentation: " + str(opt.data_augmentation) + "\n")
        config_txt_file.write("Learn rate: " + str(opt.lr) + "\n")
        config_txt_file.write("Learn mode: " + opt.learn_mode + "\n")  
        if opt.learn_mode == "SPL":
            config_txt_file.write("SPL mode: " + opt.spl_mode + "\n")
            config_txt_file.write("The parameter of the Training Scheduling: " + str(opt.r) + "\n")
        config_txt_file.close()

    #-------------------------------#
    #-------------------------------#
    model_input_size = (int(opt.model_input_size.split("_")[0]), int(opt.model_input_size.split("_")[1])) # H,W
    
    Cuda = True

    if opt.learn_mode == "SPL":
        annotation_root_path = "./variable_weight/"
        os.makedirs(annotation_root_path, exist_ok=True)
    elif opt.learn_mode == "HEM":
        annotation_root_path = "./variable_weight_HEM/"
        os.makedirs(annotation_root_path, exist_ok=True)
    else:
        annotation_root_path = "./dataloader/"

    train_annotation_path = annotation_root_path + "img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_" + Add_name + "_train.txt"
    if not os.path.exists(train_annotation_path):
        shutil.copy("./dataloader/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train.txt", train_annotation_path)

    train_dataset_image_path = opt.data_root_path + "images/train/"
    
    val_annotation_path = "./dataloader/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_val.txt"
    val_dataset_image_path = opt.data_root_path + "images/val/"
    #-------------------------------#
    # 
    #-------------------------------#
    classes_path = 'model_data/classes.txt'   
    class_names = get_classes(classes_path)
    num_classes = len(class_names) + 1 #### Include background
    
    # create model
    ### FBODInferenceBody parameters:
    ### input_img_num=5, aggregation_output_channels=16, aggregation_method="multiinput", input_mode="GRG", ### Aggreagation parameters.
    ### backbone_name="cspdarknet53": ### Extract parameters. input_channels equal to aggregation_output_channels.
    model = FBODInferenceBody(input_img_num=opt.input_img_num, aggregation_output_channels=opt.aggregation_output_channels,
                              aggregation_method=opt.aggregation_method, input_mode=opt.input_mode, backbone_name=opt.backbone_name, fusion_method=opt.fusion_method)

    #-------------------------------------------#
    #   load model
    #-------------------------------------------#
    if os.path.exists(opt.pretrain_model_path):
        print('Loading weights into state dict...')
        if Cuda:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(opt.pretrain_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Finished loading pretrained model!')
    else:
        print('Train the model from scratch!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net = net.cuda()

    # 建立loss函数
    # dynamic label assign, so the gettargets is ture.
    loss_func_train = LossFunc(num_classes=num_classes, model_input_size=(model_input_size[1], model_input_size[0]), \
                         learn_mode=opt.learn_mode, cuda=Cuda, gettargets=True)
    
    loss_func_val = LossFunc(num_classes=num_classes, model_input_size=(model_input_size[1], model_input_size[0]), \
                         learn_mode="All_Sample", cuda=Cuda, gettargets=True)


    # For calculating the AP50
    detect_post_process = FB_Postprocess(batch_size=opt.Batch_size, model_input_size=model_input_size)
    labels_to_results = LablesToResults(batch_size=opt.Batch_size)

    get_box_info_for_one_image = getBoxInfoListForOneImage_Loss(num_classes=num_classes, image_size = (model_input_size[1],model_input_size[0]),
                                                                scale=opt.scale_factor, cuda=Cuda) # image_size w,h

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
        num_train = len(train_lines)
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
        num_val = len(val_lines)


    
    #------------------------------------------------------#
    #------------------------------------------------------#
    lr = opt.lr
    Batch_size = opt.Batch_size
    start_Epoch = opt.start_Epoch
    lr = lr*((0.95)**start_Epoch)
    end_Epoch = opt.end_Epoch

    optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
    
    train_data = CustomDataset(train_lines, (model_input_size[1], model_input_size[0]), image_path=train_dataset_image_path, \
                               input_mode=opt.input_mode, continues_num=opt.input_img_num, data_augmentation=opt.data_augmentation)
    train_dataloader = DataLoader(train_data, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset_collate)
    
    val_data = CustomDataset(val_lines, (model_input_size[1], model_input_size[0]), image_path=val_dataset_image_path, \
                             input_mode=opt.input_mode, continues_num=opt.input_img_num)
    val_dataloader = DataLoader(val_data, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset_collate)


    epoch_size = max(1, num_train//Batch_size)
    epoch_size_val = num_val//Batch_size
    

    largest_AP_50=0
    for epoch in range(start_Epoch,end_Epoch):
        if opt.learn_mode == "SPL" or opt.learn_mode == "HEM":
            net = net.eval()
            print("Update Object Weight")
            image_info_list = []
            with tqdm(total=len(train_lines)) as pbar:
                for line in train_lines:
                    images, raw_bboxes, bboxes, first_img_name = load_data_raw_resize_boxes(line, train_dataset_image_path, frame_num=opt.input_img_num, image_size=model_input_size)
                    image_info_instance = image_info(iname=first_img_name)
                    if len(bboxes) == 0:
                        image_info_list.append(image_info_instance)
                        pbar.update(1)
                        continue
                    _, model_input= GetMiddleImg_ModelInput_for_MatImageList(images, model_input_size=model_input_size, continus_num=opt.input_img_num, input_mode=opt.input_mode)
                    with torch.no_grad():
                        model_input = torch.from_numpy(model_input)
                        if Cuda:
                            model_input = model_input.cuda()
                        predictions = net(model_input)
                    ### sample loss
                    image_info_instance.box_info_list = get_box_info_for_one_image(predictions, raw_bboxes, bboxes)
                    image_info_list.append(image_info_instance)
                    pbar.update(1)
            

            ### Update the object weight by rewriting all the information.
            annotation_file = open(train_annotation_path,'w')
            if opt.learn_mode == "SPL":
                spl_threshold = adjust_spl_threshold(step_proportion=(epoch*1.)/end_Epoch, r=opt.r)
                for image_info_instance in image_info_list:
                    annotation_file.write(image_info_instance.iname)
                    if len(image_info_instance.box_info_list) == 0:
                        annotation_file.write(" None")
                    else:
                        for box_info_instance in image_info_instance.box_info_list:
                            if opt.spl_mode == "hard":
                                sample_weight = spl_sampleWeight_Hard(sample_loss=box_info_instance.sample_loss, spl_threshold=spl_threshold)
                            elif opt.spl_mode == "linear":
                                sample_weight = spl_sampleWeight_Linear(sample_loss=box_info_instance.sample_loss, spl_threshold=spl_threshold)
                            elif opt.spl_mode == "logarithmic":
                                sample_weight = spl_sampleWeight_Logarithmic(sample_loss=box_info_instance.sample_loss, spl_threshold=spl_threshold)
                            elif opt.spl_mode == "Polynomial":
                                sample_weight = spl_sampleWeight_Polynomial(sample_loss=box_info_instance.sample_loss, spl_threshold=spl_threshold)
                            elif opt.spl_mode == "Continuous":
                                sample_weight = spl_sampleWeight_Continuous(sample_loss=box_info_instance.sample_loss, spl_threshold=spl_threshold)
                            else:
                                raise("Error, no such spl mode.")
                            string_label = " " + ",".join(str(int(a)) for a in box_info_instance.bbox) + "," + str(int(box_info_instance.class_id)) + "," + str(sample_weight)
                            annotation_file.write(string_label)
                    annotation_file.write("\n")
            elif opt.learn_mode == "HEM":
                sample_loss_list = []
                for image_info_instance in image_info_list:
                    if len(image_info_instance.box_info_list) == 0:
                        continue
                    else:
                        for box_info_instance in image_info_instance.box_info_list:
                            sample_loss_list.append(box_info_instance.sample_loss)
                sample_loss_list.sort(reverse=True)
                loss_threshold = sample_loss_list[int(len(sample_loss_list) * 0.4)]

                for image_info_instance in image_info_list:
                    annotation_file.write(image_info_instance.iname)
                    if len(image_info_instance.box_info_list) == 0:
                        annotation_file.write(" None")
                    else:
                        for box_info_instance in image_info_instance.box_info_list:
                            if box_info_instance.sample_loss >= loss_threshold:
                                sample_weight = 1
                            else:
                                sample_weight = 0
                            string_label = " " + ",".join(str(int(a)) for a in box_info_instance.bbox) + "," + str(int(box_info_instance.class_id)) + "," + str(sample_weight)
                            annotation_file.write(string_label)
                    annotation_file.write("\n")
            annotation_file.close()
            
            with open(train_annotation_path) as f:
                train_lines = f.readlines()
            train_data = CustomDataset(train_lines, (model_input_size[1], model_input_size[0]), image_path=train_dataset_image_path, \
                                    input_mode=opt.input_mode, continues_num=opt.input_img_num, data_augmentation=opt.data_augmentation)
            train_dataloader = DataLoader(train_data, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset_collate)

            net = net.train()

        train_loss, val_loss,largest_AP_50_record, AP_50 = fit_one_epoch(largest_AP_50,net,loss_func_train,loss_func_val,epoch,epoch_size,epoch_size_val,train_dataloader,val_dataloader,
                                                                            end_Epoch,Cuda,save_model_dir, labels_to_results=labels_to_results, detect_post_process=detect_post_process)
        largest_AP_50 = largest_AP_50_record
        if (epoch+1)>=2:
            draw_curve_loss(epoch+1, train_loss.item(), val_loss.item(), log_pic_name_loss)
        if (epoch+1)>=30:
            draw_curve_ap50(epoch+1, AP_50, log_pic_name_ap50)
        lr_scheduler.step()
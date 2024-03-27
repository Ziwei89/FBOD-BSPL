import torch
import torch.nn as nn
import sys
import math
import numpy as np
sys.path.append("..")
from .getDynamicTargets import getTargets

def MSELoss(pred,target):
    return (pred-target)**2

def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area,min = 1e-6)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)
    
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    # 计算对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh,2), axis=-1)
    ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal,min = 1e-6)
    
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0]/torch.clamp(b1_wh[..., 1],min = 1e-6)) - torch.atan(b2_wh[..., 0]/torch.clamp(b2_wh[..., 1],min = 1e-6))), 2)
    alpha = v / torch.clamp((1.0 - iou + v),min=1e-6)
    ciou = ciou - alpha * v
    return ciou

    
class Box_info(object):
    def __init__(self, box_id, bbox, xyxy = False):
        ### bbox: bbox[0] cx, bbox[1] cy, bbox[2] w, bbox[3] h, bbox[4] class_id, bbox[5] object score (difficult)
        self.__box_id = box_id
        self.__class_id = bbox[4]
        if xyxy==False:
            min_x = bbox[0] - bbox[2]/2
            min_y = bbox[1] - bbox[3]/2

            max_x = bbox[0] + bbox[2]/2
            max_y = bbox[1] + bbox[3]/2
            self.__bbox = [min_x, min_y, max_x, max_y]
        else:
            self.__bbox = bbox[:4] # x1, y1, x2, y2
        self.__positive_points_map = None
        self.post_score = None
        self.sample_loss = None

    @property
    def box_id(self):
        return self.__box_id
    @property
    def class_id(self):
        return self.__class_id
    @property
    def bbox(self):
        return self.__bbox
    @property
    def positive_points_map(self):
        temp_positive_points_map = self.__positive_points_map
        return temp_positive_points_map
    @positive_points_map.setter
    def positive_points_map(self,positive_points_map):
        self.__positive_points_map=positive_points_map

class image_info(object):
    def __init__(self, iname):
        self.__iname = iname
        self.box_info_list = []
    @property
    def iname(self):
        return self.__iname

def is_point_in_bbox(point, bbox):
    condition1 = (point[0] >= bbox[0]-bbox[2]/2) and (point[0] <= bbox[0]+bbox[2]/2)
    condition2 = (point[1] >= bbox[1]-bbox[3]/2) and (point[1] <= bbox[1]+bbox[3]/2)
    if condition1 and condition2:
        return True
    else:
        return False

def is_point_in_ellipse(point, ellipse_parameters, guassion_variance):
    point = [point[0]-ellipse_parameters[0],point[1]-ellipse_parameters[1]]
    if ((point[0]**2)/(ellipse_parameters[2]**2) + (point[1]**2)/(ellipse_parameters[3]**2)) < 1:
        guassion_value = math.exp((-1)*(point[0]**2/(2*guassion_variance[0]**2)+ point[1]**2/(2*guassion_variance[1]**2)))
        return True, guassion_value
    else:
        return False, 0

def min_max_ref_point_index(bbox, output_feature, image_size):
    min_x = bbox[0] - bbox[2]/2
    min_y = bbox[1] - bbox[3]/2

    max_x = bbox[0] + bbox[2]/2
    max_y = bbox[1] + bbox[3]/2
    min_wight_index = math.floor(max((min_x*output_feature[0])/image_size[0] - 1/2,0))
    min_height_index = math.floor(max((min_y*output_feature[1])/image_size[1] - 1/2,0))

    max_wight_index = math.ceil(min((max_x*output_feature[0])/image_size[0] - 1/2,output_feature[0]-1))
    max_height_index = math.ceil(min((max_y*output_feature[1])/image_size[1] - 1/2,output_feature[1]-1))

    return (min_wight_index, min_height_index, max_wight_index, max_height_index)

class getBoxInfoListForOneImage(nn.Module):
    def __init__(self, image_size, stride=2, cuda=True):
        super(getBoxInfoListForOneImage, self).__init__()
        self.image_size = image_size#(672,384)#w,h
        self.out_feature_size = [self.image_size[0]/stride, self.image_size[1]/stride]
        self.cuda = cuda
    
    def forward(self, input, raw_bboxes, bboxes):
       # input is a [CONF, LOC] list with 'bs,c,h,w' format tensor.
        bs = input[0].size(0)
        if bs > 1:
            raise print("Error! Can't process multi-batch!")

        # Branch for task, there are 2 tasks, that is CONF(Conf), and LOC(LOCation).
        ################# CONF #############################
        predict_CONF = input[0] ## 1,1,h,w
        predict_CONF = torch.sigmoid(predict_CONF)
        predict_CONF = predict_CONF.squeeze() ### h,w

        box_info_list = self.__get_boxes_info(raw_bboxes=raw_bboxes, bboxes=bboxes)
        for box_info in box_info_list:
            predict_conf = predict_CONF* box_info.positive_points_map
            predict_conf = torch.max(predict_conf)
            box_info.post_score = float(predict_conf.cpu().detach().numpy())
        return box_info_list
      
    def __get_boxes_info(self, raw_bboxes, bboxes): ###
        ###  bbox[0] x1, bbox[1] y1, bbox[2] x2, bbox[3] y2, bbox[4] class_id, bbox[5] object score (difficult) ###
        ###  raw_bbox[0] x1, raw_bbox[1] y1, raw_bbox[2] x2, raw_bbox[3] y2, raw_bbox[4] class_id, raw_bbox[5] object score (difficult) ###
        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        if len(bboxes) == 0:
            return None
        # convert x1,y1,x2,y2 to cx,cy,o_w,o_h
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] ## o_w
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] ## o_h
        bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2 # cx
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2 # cy

        box_id_list = []
        box_info_list = []
        box_ids_map = np.array([-1.]*int(self.out_feature_size[0])*int(self.out_feature_size[1]))

        sample_position_list = []

        for box_id, (raw_bbox, bbox) in enumerate(zip(raw_bboxes, bboxes)):
            obj_area = bbox[2] * bbox[3]
            if obj_area == 0:
                continue
            box_id_list.append(box_id)
            ###  bbox[0] cx, bbox[1] cy, bbox[2] o_w, bbox[3] o_h, bbox[4] class_id, bbox[5] difficult ###
            box_info_list.append(Box_info(box_id=box_id, bbox=raw_bbox, xyxy=True))

            min_wight_index, min_height_index, max_wight_index, max_height_index = min_max_ref_point_index(bbox,self.out_feature_size,self.image_size)
            for i in range(min_height_index, max_height_index+1):
                for j in range(min_wight_index, max_wight_index+1):
                    ref_point_position = []
                    ref_point_position.append(j*(self.image_size[0]/self.out_feature_size[0]) + (self.image_size[0]/self.out_feature_size[0])/2) #### x
                    ref_point_position.append(i*(self.image_size[1]/self.out_feature_size[1]) + (self.image_size[1]/self.out_feature_size[1])/2) #### y

                    if is_point_in_bbox(ref_point_position, bbox[:4]):# The point is in bbox.
                        if (i,j) in sample_position_list:
                            box_ids_map[i*int(self.out_feature_size[0]) + j] = -1 # # Ignore this point
                        else:
                            sample_position_list.append((i,j))
                            box_ids_map[i*int(self.out_feature_size[0]) + j] = box_id # box_id

        for box_id in box_id_list:
            True_False_map = box_ids_map == box_id
            if any(True_False_map): # If any member of box_id_map is True, return True; all the member is False, return False.
                for box_info in box_info_list:
                    if box_info.box_id == box_id:
                        True_False_map = True_False_map.reshape(int(self.out_feature_size[1]), int(self.out_feature_size[0])) ### h,w
                        True_False_map = torch.from_numpy(True_False_map)
                        True_False_map = True_False_map.type(FloatTensor)
                        box_info.positive_points_map = True_False_map
            else:
                for box_info in box_info_list:
                    if box_info.box_id == box_id:
                        box_info_list.remove(box_info)
        return box_info_list

class LossFuncForOneImage(nn.Module): #
    def __init__(self,num_classes, model_input_size=(672,384), scale=80., stride=2, cuda=True):
        super(LossFuncForOneImage, self).__init__()
        self.num_classes = num_classes
        self.model_input_size = model_input_size
        self.scale = scale
        #(model_input_size, num_classes=2, stride=2)
        self.get_targets = getTargets(model_input_size, num_classes, scale=scale, stride=stride, cuda=True)
        self.cuda = cuda
    
    def forward(self, input, bboxes):

        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        # bboxes, bbox[0] cx, bbox[1] cy, bbox[2] w, bbox[3] h, bbox[4] class_id, bbox[5] score
        targets = self.get_targets(input, bboxes, difficult_mode=0) ### targets is a list wiht 2 members, each is a 'bs,in_h,in_w,c' format tensor(cls and bbox).

        # input is a list with with 2 members(CONF and LOC), each member is a 'bs,c,in_h,in_w' format tensor).
        bs = input[0].size(0)
        if bs > 1:
            raise print("Error! Can't process multi-batch!")
        in_h = input[0].size(2) # in_h = model_input_size[1]/stride (stride = 2)
        in_w = input[0].size(3) # in_w

        # 2,bs,c,in_h,in_w -> 2,bs,in_h,in_w,c (a list with 2 members, each member is a 'bs,in_h,in_w,c' format tensor).

        # Branch for task, there are 2 tasks, that is CONF(CONFidence), and LOC(LOCation).
        # To get 3D tensor 'bs, in_h, in_w' or 4D tensor 'bs, in_h, in_w, c'.
        #################CONF
        # 
        predict_CONF = input[0].type(FloatTensor) #bs,c,in_h,in_w  c=1
        predict_CONF = predict_CONF.view(bs,in_h,in_w) #bs,in_h,in_w
        ### bs, in_h, in_w
        predict_CONF = torch.sigmoid(predict_CONF)


        #################LOC

        # bs, in_h, in_w
        ref_point_xs = ((torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1))*(self.model_input_size[0]/in_w) + (self.model_input_size[0]/in_w)/2).repeat(bs, 1, 1)
        ref_point_xs = ref_point_xs.type(FloatTensor)

        # bs, in_w, in_h
        ref_point_ys = ((torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1))*(self.model_input_size[1]/in_h) + (self.model_input_size[1]/in_h)/2).repeat(bs, 1, 1)
        # bs, in_w, in_h -> bs, in_h, in_w
        ref_point_ys = ref_point_ys.permute(0, 2, 1).contiguous()#
        ref_point_ys = ref_point_ys.type(FloatTensor)

        predict_LOC = input[1].type(FloatTensor) #bs, c,in_h,in_w  c=4(dx1,dy1,dx2,dy2)
        # bs, c, in_h, in_w -> bs, in_h, in_w, c
        predict_LOC = predict_LOC.permute(0, 2, 3, 1).contiguous()
        # Decode boxes (x1,y1,x2,y2)
        
        predict_LOC[..., 0] = predict_LOC[..., 0]*self.scale + ref_point_xs
        predict_LOC[..., 1] = predict_LOC[..., 1]*self.scale + ref_point_ys
        predict_LOC[..., 2] = predict_LOC[..., 2]*self.scale + ref_point_xs
        predict_LOC[..., 3] = predict_LOC[..., 3]*self.scale + ref_point_ys

        ### bs, in_h, in_w, c(c=4)
        ### (x1,y1,x2,y2) ----->  (cx,cy,o_w,o_h)
        predict_LOC[..., 2] = predict_LOC[..., 2] - predict_LOC[..., 0]
        predict_LOC[..., 3] = predict_LOC[..., 3] - predict_LOC[..., 1]
        predict_LOC[..., 0] = predict_LOC[..., 0] + predict_LOC[..., 2]/2
        predict_LOC[..., 1] = predict_LOC[..., 1] + predict_LOC[..., 3]/2
        ###########################

        # targets is a list wiht 2 members, each is a 'bs*in_h,in_w*c' format tensor(cls and bbox).
        # 2,bs*c*in_h,in_w -> 3,bs,in_h,in_w,c (a list with 3 members, each member is a 'bs,in_h,in_w,c' format tensor).

        #################CONF_CLS
        ### bs, in_h, in_w, c(c=num_classes(Include background))
        label_CONF_CLS = targets[0].type(FloatTensor) #bs*in_h,in_w*c  c=num_classes(Include background)
        label_CONF_CLS = label_CONF_CLS.view(bs,in_h,in_w,-1) # bs,in_h,in_w,c
        ### bs, in_h, in_w
        label_CONF = torch.sum(label_CONF_CLS[:,:,:,1:], dim=3) # bs, in_h, in_w ## Conf
        label_CLS_weight =  torch.ceil(label_CONF_CLS) # bs,in_h,in_w,c
        weight_neg = label_CLS_weight[:,:,:,:1] # bs,in_h,in_w,c(c = 1)
        if self.num_classes > 2:
            weight_non_ignore = torch.sum(label_CLS_weight,3).unsqueeze(3)
            weight_pos = (1 - weight_neg)*weight_non_ignore # Exclude rows with all zeros.
        else:
            weight_pos = label_CLS_weight[:,:,:,1:] # bs,in_h,in_w,c(c = 1)

        ### bs,in_h,in_w
        weight_pos = weight_pos.squeeze(3)

        #################LOC
        label_LOC_difficult_lamda = targets[1].type(FloatTensor) #bs*in_h,in_w*c  c=6(cx,xy,o_w,o_h,difficult,lamda)
        label_LOC_difficult_lamda = label_LOC_difficult_lamda.view(bs,in_h,in_w,-1) # bs,in_h,in_w,c(c=6)
        ### bs, in_h, in_w, c(c=4 cx,xy,o_w,o_h)
        label_LOC = label_LOC_difficult_lamda[:,:,:,:4] # bs,in_h,in_w,c(c=4)

        ## Conf Loss
        ## bs, in_h, in_w
        CONF_loss_map = MSELoss(label_CONF, predict_CONF)
        
        ### Locate Loss
        ciou_loss = 1-box_ciou(predict_LOC, label_LOC)
        ###(bs, in_h, in_w)
        LOC_loss_map = (ciou_loss.view(bs,in_h,in_w)) * weight_pos

        loss_map = (CONF_loss_map + LOC_loss_map/2)/2  ## To guaranteed that the loss of each anchor point ranges from 0 to 1.
        return loss_map


class getBoxInfoListForOneImage_Loss(nn.Module):
    def __init__(self, num_classes, image_size, scale=80., stride=2, cuda=True):
        super(getBoxInfoListForOneImage_Loss, self).__init__()
        self.image_size = image_size#(672,384)#w,h
        self.out_feature_size = [self.image_size[0]/stride, self.image_size[1]/stride]
        self.cuda = cuda
        self.loss_func_one_image = LossFuncForOneImage(num_classes=num_classes, model_input_size=image_size, scale=scale, stride=stride, cuda=cuda)
    
    def forward(self, input, raw_bboxes, bboxes):
       # input is a [CONF, LOC] list with 'bs,c,h,w' format tensor.
        bs = input[0].size(0)
        if bs > 1:
            raise print("Error! Can't process multi-batch!")
        
        loss_one_image_map = self.loss_func_one_image(input, bboxes) ## 1,h,w
        loss_one_image_map = loss_one_image_map.squeeze() ### h,w

        box_info_list = self.__get_boxes_info(raw_bboxes=raw_bboxes, bboxes=bboxes)
        for box_info in box_info_list:
            loss_one_image_one_box_map = loss_one_image_map * box_info.positive_points_map
            loss_one_image_one_box = torch.sum(loss_one_image_one_box_map)/torch.sum(box_info.positive_points_map)
            box_info.sample_loss = float(loss_one_image_one_box.cpu().detach().numpy())
        return box_info_list
      
    def __get_boxes_info(self, raw_bboxes, bboxes): ###
        ###  bbox[0] x1, bbox[1] y1, bbox[2] x2, bbox[3] y2, bbox[4] class_id, bbox[5] object score (difficult) ###
        ###  raw_bbox[0] x1, raw_bbox[1] y1, raw_bbox[2] x2, raw_bbox[3] y2, raw_bbox[4] class_id, raw_bbox[5] object score (difficult) ###
        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        if len(bboxes) == 0:
            return None
        # convert x1,y1,x2,y2 to cx,cy,o_w,o_h
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] ## o_w
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] ## o_h
        bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2 # cx
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2 # cy

        box_id_list = []
        box_info_list = []
        box_ids_map = np.array([-1.]*int(self.out_feature_size[0])*int(self.out_feature_size[1]))

        sample_position_list = []

        for box_id, (raw_bbox, bbox) in enumerate(zip(raw_bboxes, bboxes)):
            obj_area = bbox[2] * bbox[3]
            if obj_area == 0:
                continue
            box_id_list.append(box_id)
            ###  bbox[0] cx, bbox[1] cy, bbox[2] o_w, bbox[3] o_h, bbox[4] class_id, bbox[5] difficult ###
            box_info_list.append(Box_info(box_id=box_id, bbox=raw_bbox, xyxy=True))

            min_wight_index, min_height_index, max_wight_index, max_height_index = min_max_ref_point_index(bbox,self.out_feature_size,self.image_size)
            for i in range(min_height_index, max_height_index+1):
                for j in range(min_wight_index, max_wight_index+1):
                    ref_point_position = []
                    ref_point_position.append(j*(self.image_size[0]/self.out_feature_size[0]) + (self.image_size[0]/self.out_feature_size[0])/2) #### x
                    ref_point_position.append(i*(self.image_size[1]/self.out_feature_size[1]) + (self.image_size[1]/self.out_feature_size[1])/2) #### y

                    if is_point_in_bbox(ref_point_position, bbox[:4]):# The point is in bbox.
                        if (i,j) in sample_position_list:
                            box_ids_map[i*int(self.out_feature_size[0]) + j] = -1 # # Ignore this point
                        else:
                            sample_position_list.append((i,j))
                            box_ids_map[i*int(self.out_feature_size[0]) + j] = box_id # box_id

        for box_id in box_id_list:
            True_False_map = box_ids_map == box_id
            if any(True_False_map): # If any member of box_id_map is True, return True; all the member is False, return False.
                for box_info in box_info_list:
                    if box_info.box_id == box_id:
                        True_False_map = True_False_map.reshape(int(self.out_feature_size[1]), int(self.out_feature_size[0])) ### h,w
                        True_False_map = torch.from_numpy(True_False_map)
                        True_False_map = True_False_map.type(FloatTensor)
                        box_info.positive_points_map = True_False_map
            else:
                for box_info in box_info_list:
                    if box_info.box_id == box_id:
                        box_info_list.remove(box_info)
        return box_info_list

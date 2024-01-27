import torch
import torch.nn as nn
import math
import numpy as np
import copy

def boxes_iou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(..., 4), xywh
    b2: tensor, shape=(..., 4), xywh

    返回为：
    -------
    iou: tensor, shape=(..., 1)
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
    return iou

class Box_info(object):
    def __init__(self, box_id, bbox, xyxy = False):
        ### bbox: bbox[0] cx, bbox[1] cy, bbox[2] w, bbox[3] h, bbox[4] class_id, bbox[5] object score (difficult)
        self.__box_id = box_id
        if xyxy==False:
            min_x = bbox[0] - bbox[2]/2
            min_y = bbox[1] - bbox[3]/2

            max_x = bbox[0] + bbox[2]/2
            max_y = bbox[1] + bbox[3]/2
            self.__bbox = [min_x, min_y, max_x, max_y]
        else:
            self.__bbox = bbox[:4] # x1, y1, x2, y2
        self.__positive_points_map = None

    @property
    def box_id(self):
        return self.__box_id
    @property
    def bbox(self):
        return self.__bbox
    @property
    def positive_points_map(self):
        temp_positive_points_map = self.__positive_points_map
        # self.__positive_points_map=None
        return temp_positive_points_map
    @positive_points_map.setter
    def positive_points_map(self,positive_points_map):
        self.__positive_points_map=positive_points_map

def is_point_in_ellipse(point, ellipse_parameters, guassion_variance):
    point = [point[0]-ellipse_parameters[0],point[1]-ellipse_parameters[1]]
    if ((point[0]**2)/(ellipse_parameters[2]**2) + (point[1]**2)/(ellipse_parameters[3]**2)) < 1:
        guassion_value = math.exp((-1)*(point[0]**2/(2*guassion_variance[0]**2)+ point[1]**2/(2*guassion_variance[1]**2)))
        return True, guassion_value
    else:
        return False, 0

def is_point_in_bbox(point, bbox):
    condition1 = (point[0] >= bbox[0]-bbox[2]/2) and (point[0] <= bbox[0]+bbox[2]/2)
    condition2 = (point[1] >= bbox[1]-bbox[3]/2) and (point[1] <= bbox[1]+bbox[3]/2)
    if condition1 and condition2:
        return True
    else:
        return False

def min_max_ref_point_index(bbox, output_feature, model_input_size):
    min_x = bbox[0] - bbox[2]/2
    min_y = bbox[1] - bbox[3]/2

    max_x = bbox[0] + bbox[2]/2
    max_y = bbox[1] + bbox[3]/2
    min_wight_index = math.floor(max((min_x*output_feature[0])/model_input_size[0] - 1/2,0))
    min_height_index = math.floor(max((min_y*output_feature[1])/model_input_size[1] - 1/2,0))

    max_wight_index = math.ceil(min((max_x*output_feature[0])/model_input_size[0] - 1/2,output_feature[0]-1))
    max_height_index = math.ceil(min((max_y*output_feature[1])/model_input_size[1] - 1/2,output_feature[1]-1))

    return (min_wight_index, min_height_index, max_wight_index, max_height_index)

def getSPL_SampleWeight_hard(sample_score, threshold_lamda):
    if sample_score >= threshold_lamda:
        return 1
    else:
        return 0

def getSPL_SampleWeight_soft(sample_score, m_root, threshold_lamda):
    if sample_score >= threshold_lamda:
        return sample_score ** (1/m_root)
    else:
        return 0

class getTargetsWithSPL_SampleWeight(nn.Module):
    def __init__(self, model_input_size, num_classes=2, scale=80., stride=2, assign_method="auto_assign", default_inner_proportion=0.7, default_guassion_value=0.3, m_root=6, cuda=True):
        super(getTargetsWithSPL_SampleWeight, self).__init__()
        self.model_input_size = model_input_size#(672,384)#feature_w,feature_h
        self.num_classes = num_classes
        self.scale = scale
        self.assign_method = assign_method
        self.default_inner_proportion = default_inner_proportion
        self.default_guassion_value = default_guassion_value
        self.out_feature_size = [self.model_input_size[0]/stride, self.model_input_size[1]/stride]
        self.size_per_ref_point = self.model_input_size[0]/self.out_feature_size[0]
        self.m_root = m_root
        self.cuda = cuda

    def forward(self, input, bboxes_bs, threshold_lamda):
        # input is a [CONF, LOC] list with 'bs,c,h,w' format tensor.
        # bboxes is a bs list with 'n,c' tensor, n is the num of box.
        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        targets = [] ### targets is a list wiht 2 members, each is a 'bs,h,w,c' format tensor(cls and bbox).
        targets_cls = []
        targets_loc = []

        bs = input[0].size(0)
        in_h = input[0].size(2) # in_h = model_input_size[1]/2 (stride = 2)
        in_w = input[0].size(3) # in_w

        predict_CONF = input[0] # 'bs,1,h,w' format tensor, c=1
        predict_CONF = predict_CONF.squeeze() # 'bs,h,w' format tensor

        if self.assign_method == "auto_assign":
            ################# Get predict bboxes

            # bs, h, w
            ref_point_xs = ((torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1))*(self.model_input_size[0]/in_w) + (self.model_input_size[0]/in_w)/2).repeat(bs, 1, 1)
            ref_point_xs = ref_point_xs.type(FloatTensor)

            # bs, w, h
            ref_point_ys = ((torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1))*(self.model_input_size[1]/in_h) + (self.model_input_size[1]/in_h)/2).repeat(bs, 1, 1)
            # bs, w, h -> bs, h, w
            ref_point_ys = ref_point_ys.permute(0, 2, 1).contiguous()#
            ref_point_ys = ref_point_ys.type(FloatTensor)

            predict_bboxes_bs = input[1] #bs, c,h,w  c=4(dx1,dy1,dx2,dy2)
            # bs, c, h, w -> bs, h, w, c
            predict_bboxes_bs = predict_bboxes_bs.permute(0, 2, 3, 1).contiguous()
            # Decode boxes (x1,y1,x2,y2)
            
            predict_bboxes_bs[..., 0] = predict_bboxes_bs[..., 0]*self.scale + ref_point_xs
            predict_bboxes_bs[..., 1] = predict_bboxes_bs[..., 1]*self.scale + ref_point_ys
            predict_bboxes_bs[..., 2] = predict_bboxes_bs[..., 2]*self.scale + ref_point_xs
            predict_bboxes_bs[..., 3] = predict_bboxes_bs[..., 3]*self.scale + ref_point_ys

            ### bs, h, w, c(c=4)
            ### (x1,y1,x2,y2) ----->  (cx,cy,o_w,o_h)
            predict_bboxes_bs[..., 2] = predict_bboxes_bs[..., 2] - predict_bboxes_bs[..., 0]
            predict_bboxes_bs[..., 3] = predict_bboxes_bs[..., 3] - predict_bboxes_bs[..., 1]
            predict_bboxes_bs[..., 0] = predict_bboxes_bs[..., 0] + predict_bboxes_bs[..., 2]/2
            predict_bboxes_bs[..., 1] = predict_bboxes_bs[..., 1] + predict_bboxes_bs[..., 3]/2
            ###########################

        for b in range(bs):
            predict_CONF_one_batch = predict_CONF[b] # 'h,w' format tensor
            predict_CONF_one_batch = torch.sigmoid(predict_CONF_one_batch)

            bboxes = bboxes_bs[b]
            if self.assign_method == "binary_assign":
                box_info_list, label_list = self.__get_boxes_info_and_targets(bboxes=bboxes) ### label_list[0], label_list[1] '1,h,w,c'
            elif self.assign_method == "guassian_assign":
                box_info_list, label_list = self.__get_boxes_info_and_targets_with_guassion(bboxes=bboxes) ### label_list[0], label_list[1] '1,h,w,c'
            elif self.assign_method == "auto_assign":
                predict_bboxes = predict_bboxes_bs[b]
                box_info_list, label_list = self.__get_boxes_info_and_targets_with_dynamicLableAssign(predict_bbox=predict_bboxes, bboxes=bboxes) ### label_list[0], label_list[1] '1,h,w,c'
            else:
                raise("Error!, assign_method error.")
            
            if box_info_list == None:
                targets_cls.append(label_list[0])
                targets_loc.append(label_list[1])
                continue
            sample_weight_map = np.array([0.]*int(self.out_feature_size[0])*int(self.out_feature_size[1]))
            sample_weight_map = sample_weight_map.reshape(int(self.out_feature_size[1]), int(self.out_feature_size[0])) ## h,w
            sample_weight_map = torch.from_numpy(sample_weight_map)
            sample_weight_map = sample_weight_map.type(FloatTensor)

            for box_info in box_info_list:
                predict_conf = predict_CONF_one_batch * box_info.positive_points_map
                predict_score = torch.max(predict_conf)
                # Convert the predict_score to sample weight through SPL regularizer
                # sample_weight = getSPL_SampleWeight_soft(predict_score, self.m_root, threshold_lamda)
                sample_weight = getSPL_SampleWeight_hard(predict_score, threshold_lamda)

                sample_weight_map_for_one_box = sample_weight * box_info.positive_points_map
                sample_weight_map += sample_weight_map_for_one_box
            sample_weight_map = sample_weight_map.reshape((1, int(self.out_feature_size[1]), int(self.out_feature_size[0])))

            label_list[1][:,:,:,4] = sample_weight_map
            targets_cls.append(label_list[0])
            targets_loc.append(label_list[1])
        targets_cls = torch.cat(targets_cls, 0) ### 'bs,h,w,c' format tensor
        targets_loc = torch.cat(targets_loc, 0) ### 'bs,h,w,c' format tensor
        targets.append(targets_cls)
        targets.append(targets_loc)
        return targets

        
    def __get_boxes_info_and_targets_with_guassion(self, bboxes): ###
        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        ###  bbox[0] x1, bbox[1] y1, bbox[2] x2, bbox[3] y2, bbox[4] class_id, bbox[5] object score (difficult) ###
        label_list=[]
        class_label_map = np.array(([1.] + [0.] * (self.num_classes - 1))*int(self.out_feature_size[0])*int(self.out_feature_size[1])) ### For targets
        points_label_map = np.array([1.]*6*int(self.out_feature_size[0])*int(self.out_feature_size[1])) ### For targets
        if len(bboxes) == 0:

            class_label_map = class_label_map.reshape(int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### h,w,c
            points_label_map = points_label_map.reshape(int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### h,w,c

            class_label_map = torch.from_numpy(class_label_map)
            points_label_map = torch.from_numpy(points_label_map)

            label_list.append(class_label_map)
            label_list.append(points_label_map)
            return None, label_list
        # convert x1,y1,x2,y2 to cx,cy,o_w,o_h
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] ## o_w
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] ## o_h
        bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2 # cx
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2 # cy

        box_id_list = []
        box_info_list = []
        box_ids_map = np.array([-1.]*int(self.out_feature_size[0])*int(self.out_feature_size[1]))
        position_guassian_value_dic ={}

        for box_id, bbox in enumerate(bboxes):
            obj_area = bbox[2] * bbox[3]
            if obj_area == 0:
                continue
            box_id_list.append(box_id)
            box_info_list.append(Box_info(box_id=box_id, bbox=bbox))
            length_x_semi_axis = bbox[2]/2
            length_y_semi_axis = bbox[3]/2

            length_inner_ellipse_x_semi_axis = length_x_semi_axis * self.default_inner_proportion
            length_inner_ellipse_y_semi_axis = length_y_semi_axis * self.default_inner_proportion

            ellipse_parameters = [bbox[0],bbox[1], length_inner_ellipse_x_semi_axis,length_inner_ellipse_y_semi_axis]

            guassion_variance_x = (length_inner_ellipse_x_semi_axis**2/(2*(-1)*math.log(self.default_guassion_value)))**0.5
            guassion_variance_y = (length_inner_ellipse_y_semi_axis**2/(2*(-1)*math.log(self.default_guassion_value)))**0.5

            lamda =  self.size_per_ref_point / obj_area**0.5  # Note: This parameter is not used in this version. #This parameter is related to the size of the target, and the smaller the target, the larger the parameter. 

            min_wight_index, min_height_index, max_wight_index, max_height_index = min_max_ref_point_index(bbox,self.out_feature_size,self.model_input_size)
            for i in range(min_height_index, max_height_index+1):
                for j in range(min_wight_index, max_wight_index+1):
                    ref_point_position = []
                    ref_point_position.append(j*(self.model_input_size[0]/self.out_feature_size[0]) + (self.model_input_size[0]/self.out_feature_size[0])/2) #### x
                    ref_point_position.append(i*(self.model_input_size[1]/self.out_feature_size[1]) + (self.model_input_size[1]/self.out_feature_size[1])/2) #### y

                    result = is_point_in_ellipse(ref_point_position, ellipse_parameters, [guassion_variance_x,guassion_variance_y])
                    if result[0]:# The point is in inner ellipse.
                        ### If the position has two guassian values, restore the larger one.
                        if (i,j) in position_guassian_value_dic:
                            if position_guassian_value_dic[(i,j)] >= result[1]: ### Maybe error
                                continue
                            else:
                                position_guassian_value_dic[(i,j)] = result[1]
                        else:
                            position_guassian_value_dic[(i,j)] = result[1]
                        box_ids_map[i*int(self.out_feature_size[0]) + j] = box_id # box_id

                        class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + 0] = 0 ### For targets, label
                        class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + int(bbox[4]) + 1] = result[1] ### For targets, label

                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 0] = bbox[0] # cx ### For targets, loc
                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 1] = bbox[1] # cy ### For targets, loc
                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 2] = bbox[2] # o_w ### For targets, loc
                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 3] = bbox[3] # o_h ### For targets, loc
                        

                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 4] = bbox[5] # object score
                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 5] = lamda # lamda parameter
                    else:# The point is out inner ellipse.
                        if is_point_in_bbox(ref_point_position, bbox):# The point is in object bounding box but out inner ellipse.
                            class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + 0] = 0
        class_label_map = class_label_map.reshape(1, int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### 1,h,w,c
        points_label_map = points_label_map.reshape(1, int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### 1,h,w,c
        class_label_map = torch.from_numpy(class_label_map)
        points_label_map = torch.from_numpy(points_label_map)
        label_list.append(class_label_map)
        label_list.append(points_label_map)

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
        return box_info_list, label_list

      
    def __get_boxes_info_and_targets_with_dynamicLableAssign(self, predict_bbox, bboxes): ###
        ### predict_bbox feature_h, feature_w, 4 (4: cx, cy, o_w, o_h)
        ### bboxes m,6 (6: x1, y1, x2, y2, class_id, object score)
        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        label_list=[]
        class_label_map = np.array(([1.] + [0.] * (self.num_classes - 1))*int(self.out_feature_size[0])*int(self.out_feature_size[1])) ### For targets
        points_label_map = np.array([1.]*6*int(self.out_feature_size[0])*int(self.out_feature_size[1])) ### For targets
        if len(bboxes) == 0:

            class_label_map = class_label_map.reshape(1, int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### h,w,c
            points_label_map = points_label_map.reshape(1, int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### h,w,c

            class_label_map = torch.from_numpy(class_label_map)
            points_label_map = torch.from_numpy(points_label_map)

            label_list.append(class_label_map)
            label_list.append(points_label_map)
            return None, label_list

        box_id_list = []
        box_info_list = []
        position_ciou_value_dic = {}

        # convert x1,y1,x2,y2 to cx,cy,o_w,o_h
        ### bboxes m,6 (6:cx, cy, w, h, class_id, object score)
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] ## o_w
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] ## o_h
        bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2 # cx
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2 # cy

        ### -1 means negative, or ignore, other number means box_id
        box_ids_map = np.array([-1.]*int(self.out_feature_size[0])*int(self.out_feature_size[1]))
        for box_id, bbox in enumerate(bboxes):
            obj_area = bbox[2] * bbox[3]
            if obj_area == 0:
                continue
            box_id_list.append(box_id)
            box_info_list.append(Box_info(box_id=box_id, bbox=bbox))

            ###  bbox[0] cx, bbox[1] cy, bbox[2] o_w, bbox[3] o_h, bbox[4] class_id, bbox[5] difficult ###
            bbox_position = copy.deepcopy(bbox[:4]) ###  bbox_position[0] cx, bbox_position[1] cy, bbox_position[2] o_w, bbox_position[3] o_h
            bbox_position = bbox_position.type(FloatTensor)

            # bbox_position expand 4 to feature_h, feature_w, 4
            ## self.out_feature_size[0] feature_w, self.out_feature_size[1] feature_h
            bbox_position = bbox_position.repeat(int(self.out_feature_size[0]),1).repeat(int(self.out_feature_size[1]),1,1)
            # iou: feature_h, feature_w, 1
            iou = boxes_iou(predict_bbox, bbox_position)
            # iou: feature_h, feature_w
            iou = iou.squeeze()
            if self.cuda:
                iou = iou.cpu().detach().numpy()
            else:
                iou = iou.detach().numpy()
            
            ############### 
            first_filter = np.array([0.]*int(self.out_feature_size[0])*int(self.out_feature_size[1]))
            min_wight_index, min_height_index, max_wight_index, max_height_index = min_max_ref_point_index(bbox,self.out_feature_size,self.model_input_size)
            for i in range(min_height_index, max_height_index+1):
                for j in range(min_wight_index, max_wight_index+1):
                    first_filter[i*int(self.out_feature_size[0]) + j] = 1
            first_filter = first_filter.reshape(int(self.out_feature_size[1]), int(self.out_feature_size[0]))

            iou_filter = iou * first_filter  # feature_h, feature_w  # first filter: gt box filter

            dynamic_k = np.sum(iou_filter)
            if dynamic_k < 1:
                dynamic_k = 1
            dynamic_k = math.ceil(dynamic_k)
            dropout_iou = copy.deepcopy(iou_filter)
            dropout_iou = dropout_iou.reshape(-1)
            sorted_iou = np.sort(dropout_iou)
            second_filter_index = iou_filter <= sorted_iou[-(dynamic_k+1)]
            iou_filter[second_filter_index] = 0  # second filter: dynamic_k filter
            ##############
            # The positive anchor points of the object is more, 
            # the weight(lamda) is smaller. To balance the positive anchor points number of different object.
            lamda = (1/dynamic_k)**(1/2)  # Note: This parameter is not used in this version.

            for i in range(min_height_index, max_height_index+1):
                for j in range(min_wight_index, max_wight_index+1):
                    if iou_filter[i][j] > 0:
                        if (i,j) in position_ciou_value_dic:
                            if iou_filter[i][j] < position_ciou_value_dic[(i,j)]:
                                continue
                            elif iou_filter[i][j] == position_ciou_value_dic[(i,j)]:
                                box_ids_map[i*int(self.out_feature_size[0]) + j] = -1 # # Ignore this point
                                for class_id_index in range(self.num_classes): # Ignore this point
                                    class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + class_id_index] = 0
                                continue
                            else: # ciou_filter[i][j] > position_ciou_value_dic[(i,j)]
                                position_ciou_value_dic[(i,j)] = iou_filter[i][j]
                                box_ids_map[i*int(self.out_feature_size[0]) + j] = box_id

                                for class_id_index in range(self.num_classes): # Reset the class value of this point
                                    class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + class_id_index] = 0
                        else:
                            position_ciou_value_dic[(i,j)] = iou_filter[i][j]
                            box_ids_map[i*int(self.out_feature_size[0]) + j] = box_id
                        
                        class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + 0] = 0
                        class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + int(bbox[4]) + 1] = 1

                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 0] = bbox[0] # cx
                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 1] = bbox[1] # cy
                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 2] = bbox[2] # o_w
                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 3] = bbox[3] # o_h
                        

                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 4] = bbox[5] # difficult
                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 5] = lamda # lamda parameter
                    
                    else:# The point is not in positive anchor points.
                        class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + 0] = 0

        class_label_map = class_label_map.reshape(1, int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### 1,h,w,c
        points_label_map = points_label_map.reshape(1, int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### 1,h,w,c
        class_label_map = torch.from_numpy(class_label_map)
        points_label_map = torch.from_numpy(points_label_map)
        label_list.append(class_label_map)
        label_list.append(points_label_map)
                            
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
        return box_info_list, label_list

       
    def __get_boxes_info_and_targets(self, bboxes): ###
        ###  bbox[0] x1, bbox[1] y1, bbox[2] x2, bbox[3] y2, bbox[4] class_id, bbox[5] object score (difficult) ###
        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        label_list=[]
        class_label_map = np.array(([1.] + [0.] * (self.num_classes - 1))*int(self.out_feature_size[0])*int(self.out_feature_size[1])) ### For targets
        points_label_map = np.array([1.]*6*int(self.out_feature_size[0])*int(self.out_feature_size[1])) ### For targets
        if len(bboxes) == 0:

            class_label_map = class_label_map.reshape(int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### h,w,c
            points_label_map = points_label_map.reshape(int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### h,w,c

            class_label_map = torch.from_numpy(class_label_map)
            points_label_map = torch.from_numpy(points_label_map)
            label_list.append(class_label_map)
            label_list.append(points_label_map)
            return None, label_list
        # convert x1,y1,x2,y2 to cx,cy,o_w,o_h
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] ## o_w
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] ## o_h
        bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2 # cx
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2 # cy

        box_id_list = []
        box_info_list = []
        box_ids_map = np.array([-1.]*int(self.out_feature_size[0])*int(self.out_feature_size[1]))

        sample_position_list = []

        for box_id, bbox in enumerate(bboxes):
            obj_area = bbox[2] * bbox[3]
            if obj_area == 0:
                continue
            box_id_list.append(box_id)
            box_info_list.append(Box_info(box_id=box_id, bbox=bbox))

            ###  bbox[0] cx, bbox[1] cy, bbox[2] o_w, bbox[3] o_h, bbox[4] class_id, bbox[5] difficult ###
            inner_bbox = copy.deepcopy(bbox[:4]) ###  inner_bbox[0] cx, inner_bbox[1] cy, inner_bbox[2] o_w, inner_bbox[3] o_h
            inner_bbox[2] = inner_bbox[2] * self.default_inner_proportion ### o_w
            inner_bbox[3] = inner_bbox[3] * self.default_inner_proportion ### o_h

            lamda =  self.size_per_ref_point / obj_area**0.5   # Note: This parameter is not used in this version. #This parameter is related to the size of the target, and the smaller the target, the larger the parameter. 

            min_wight_index, min_height_index, max_wight_index, max_height_index = min_max_ref_point_index(bbox,self.out_feature_size,self.model_input_size)
            for i in range(min_height_index, max_height_index+1):
                for j in range(min_wight_index, max_wight_index+1):
                    ref_point_position = []
                    ref_point_position.append(j*(self.model_input_size[0]/self.out_feature_size[0]) + (self.model_input_size[0]/self.out_feature_size[0])/2) #### x
                    ref_point_position.append(i*(self.model_input_size[1]/self.out_feature_size[1]) + (self.model_input_size[1]/self.out_feature_size[1])/2) #### y

                    if is_point_in_bbox(ref_point_position, inner_bbox):# The point is in inner bbox.
                        if (i,j) in sample_position_list:
                            box_ids_map[i*int(self.out_feature_size[0]) + j] = -1 # # Ignore this point
                            for class_id_index in range(self.num_classes): # Ignore this point
                                class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + class_id_index] = 0
                        else:
                            sample_position_list.append((i,j))
                            box_ids_map[i*int(self.out_feature_size[0]) + j] = box_id # box_id

                            class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + 0] = 0
                            class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + int(bbox[4]) + 1] = 1

                            points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 0] = bbox[0] # cx
                            points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 1] = bbox[1] # cy
                            points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 2] = bbox[2] # o_w
                            points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 3] = bbox[3] # o_h
                            

                            points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 4] = bbox[5] # difficult
                            points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 5] = lamda # lamda parameter

                    else:# The point is out inner box.
                        if is_point_in_bbox(ref_point_position, bbox):# The point is in object bounding box but out inner box.
                            class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + 0] = 0

        class_label_map = class_label_map.reshape(1, int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### 1,h,w,c
        points_label_map = points_label_map.reshape(1, int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### 1,h,w,c
        class_label_map = torch.from_numpy(class_label_map)
        points_label_map = torch.from_numpy(points_label_map)
        label_list.append(class_label_map)
        label_list.append(points_label_map)

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
        return box_info_list, label_list

class getTargets(nn.Module):
    def __init__(self, model_input_size, num_classes=2, scale=80., stride=2, cuda=True):
        super(getTargets, self).__init__()
        self.model_input_size = model_input_size#(672,384)#img_w,img_h
        self.num_classes = num_classes
        self.scale = scale
        self.out_feature_size = [self.model_input_size[0]/stride, self.model_input_size[1]/stride] ## feature_w,feature_h
        self.size_per_ref_point = self.model_input_size[0]/self.out_feature_size[0]

        self.cuda = cuda

    # def forward(self, batch_size, bboxes_bs):
    def forward(self, input, bboxes_bs, difficult_mode):
        # input is a [CONF, LOC] list with 'bs,c,h,w' format tensor.
        # bboxes is a bs list with 'n,c' tensor, n is the num of box.
        # difficult_mode # 0 means no difficult difference, 1 means simple sample only.
        targets = [] ### targets is a list wiht 2 members, each is a 'bs,h,w,c' format tensor(cls and bbox).
        targets_cls = []
        targets_loc = []


        bs = input[0].size(0)
        in_h = input[0].size(2) # in_h = model_input_size[1]/2 (stride = 2)
        in_w = input[0].size(3) # in_w

        ################# Get predict bboxes
        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # bs, h, w
        ref_point_xs = ((torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1))*(self.model_input_size[0]/in_w) + (self.model_input_size[0]/in_w)/2).repeat(bs, 1, 1)
        ref_point_xs = ref_point_xs.type(FloatTensor)

        # bs, w, h
        ref_point_ys = ((torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1))*(self.model_input_size[1]/in_h) + (self.model_input_size[1]/in_h)/2).repeat(bs, 1, 1)
        # bs, w, h -> bs, h, w
        ref_point_ys = ref_point_ys.permute(0, 2, 1).contiguous()#
        ref_point_ys = ref_point_ys.type(FloatTensor)

        predict_bboxes_bs = input[1] #bs, c,h,w  c=4(dx1,dy1,dx2,dy2)
         # bs, c, h, w -> bs, h, w, c
        predict_bboxes_bs = predict_bboxes_bs.permute(0, 2, 3, 1).contiguous()
        # Decode boxes (x1,y1,x2,y2)
        
        predict_bboxes_bs[..., 0] = predict_bboxes_bs[..., 0]*self.scale + ref_point_xs
        predict_bboxes_bs[..., 1] = predict_bboxes_bs[..., 1]*self.scale + ref_point_ys
        predict_bboxes_bs[..., 2] = predict_bboxes_bs[..., 2]*self.scale + ref_point_xs
        predict_bboxes_bs[..., 3] = predict_bboxes_bs[..., 3]*self.scale + ref_point_ys

        ### bs, h, w, c(c=4)
        ### (x1,y1,x2,y2) ----->  (cx,cy,o_w,o_h)
        predict_bboxes_bs[..., 2] = predict_bboxes_bs[..., 2] - predict_bboxes_bs[..., 0]
        predict_bboxes_bs[..., 3] = predict_bboxes_bs[..., 3] - predict_bboxes_bs[..., 1]
        predict_bboxes_bs[..., 0] = predict_bboxes_bs[..., 0] + predict_bboxes_bs[..., 2]/2
        predict_bboxes_bs[..., 1] = predict_bboxes_bs[..., 1] + predict_bboxes_bs[..., 3]/2
        ###########################

        for b in range(bs):
            bboxes = bboxes_bs[b]
            predict_bboxes = predict_bboxes_bs[b]
            label_list = self.__get_targets_with_dynamicLableAssign(predict_bbox=predict_bboxes,bboxes=bboxes, difficult_mode=difficult_mode) ### label_list[0], label_list[1] '1,h,w,c'
            targets_cls.append(label_list[0])
            targets_loc.append(label_list[1])
        targets_cls = torch.cat(targets_cls, 0) ### 'bs,h,w,c' format tensor
        targets_loc = torch.cat(targets_loc, 0) ### 'bs,h,w,c' format tensor
        targets.append(targets_cls)
        targets.append(targets_loc)
        return targets
      
    def __get_targets_with_dynamicLableAssign(self, predict_bbox, bboxes, difficult_mode): ###

        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        ### predict_bbox feature_h, feature_w, 4 (4: cx, cy, o_w, o_h)
        ### bboxes m,6 (6: x1, y1, x2, y2, class_id, object score)

        label_list=[]
        class_label_map = np.array(([1.] + [0.] * (self.num_classes - 1))*int(self.out_feature_size[0])*int(self.out_feature_size[1])) ### For targets
        points_label_map = np.array([1.]*6*int(self.out_feature_size[0])*int(self.out_feature_size[1])) ### For targets
        if len(bboxes) == 0:

            class_label_map = class_label_map.reshape(1, int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### 1,h,w,c
            points_label_map = points_label_map.reshape(1, int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### 1,h,w,c

            class_label_map = torch.from_numpy(class_label_map)
            points_label_map = torch.from_numpy(points_label_map)
            label_list.append(class_label_map)
            label_list.append(points_label_map)
            return label_list

        position_iou_value_dic = {}

        # convert x1,y1,x2,y2 to cx,cy,o_w,o_h
        ### bboxes m,6 (6:cx, cy, o_w, o_h, class_id, object score)
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] ## o_w
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] ## o_h
        bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2 # cx
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2 # cy

        for bbox in bboxes:
            obj_area = bbox[2] * bbox[3]
            if obj_area == 0:
                continue

            ###  bbox[0] cx, bbox[1] cy, bbox[2] o_w, bbox[3] o_h, bbox[4] class_id, bbox[5] difficult ###
            bbox_position = copy.deepcopy(bbox[:4]) ###  bbox_position[0] cx, bbox_position[1] cy, bbox_position[2] o_w, bbox_position[3] o_h
            bbox_position = bbox_position.type(FloatTensor)

            # bbox_position expand 4 to feature_h, feature_w, 4
            ## self.out_feature_size[0] feature_w, self.out_feature_size[1] feature_h
            bbox_position = bbox_position.repeat(int(self.out_feature_size[0]),1).repeat(int(self.out_feature_size[1]),1,1)
            # iou: feature_h, feature_w, 1
            iou = boxes_iou(predict_bbox, bbox_position)
            # iou: feature_h, feature_w
            iou = iou.squeeze()
            if self.cuda:
                iou = iou.cpu().detach().numpy()
            else:
                iou = iou.detach().numpy()
            
            ############### 
            first_filter = np.array([0.]*int(self.out_feature_size[0])*int(self.out_feature_size[1]))
            min_wight_index, min_height_index, max_wight_index, max_height_index = min_max_ref_point_index(bbox,self.out_feature_size,self.model_input_size)
            for i in range(min_height_index, max_height_index+1):
                for j in range(min_wight_index, max_wight_index+1):
                    first_filter[i*int(self.out_feature_size[0]) + j] = 1
            first_filter = first_filter.reshape(int(self.out_feature_size[1]), int(self.out_feature_size[0]))
            
            iou_filter = iou * first_filter  # feature_h, feature_w  # first filter: gt box filter

            dynamic_k = np.sum(iou_filter)
            if dynamic_k < 1:
                dynamic_k = 1
            dynamic_k = math.ceil(dynamic_k)
            dropout_iou = copy.deepcopy(iou_filter)
            dropout_iou = dropout_iou.reshape(-1)
            sorted_iou = np.sort(dropout_iou)
            second_filter_index = iou_filter <= sorted_iou[-(dynamic_k+1)]
            iou_filter[second_filter_index] = 0  # second filter: dynamic_k filter

            if difficult_mode: # simple sample only
                if bbox[5] >= 0.625:
                    object_score = 1
                else:
                    object_score = 0
            else: # no difficulty different
                object_score = 1

            ##############
            # The positive anchor points of the object is more, 
            # the weight(lamda) is smaller. To balance the positive anchor points number of different object.
            lamda = (1/dynamic_k)**(1/2)  # Note: This parameter is not used in this version.

            for i in range(min_height_index, max_height_index+1):
                for j in range(min_wight_index, max_wight_index+1):
                    if iou_filter[i][j] > 0:
                        if (i,j) in position_iou_value_dic:
                            if iou_filter[i][j] < position_iou_value_dic[(i,j)]:
                                continue
                            elif iou_filter[i][j] == position_iou_value_dic[(i,j)]:
                                for class_id_index in range(self.num_classes): # Ignore this point
                                    class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + class_id_index] = 0
                                continue
                            else: # iou_filter[i][j] > position_iou_value_dic[(i,j)]
                                position_iou_value_dic[(i,j)] = iou_filter[i][j]

                                for class_id_index in range(self.num_classes): # Reset the class value of this point
                                    class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + class_id_index] = 0
                        else:
                            position_iou_value_dic[(i,j)] = iou_filter[i][j]
                        
                        class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + 0] = 0
                        class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + int(bbox[4]) + 1] = 1

                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 0] = bbox[0] # cx
                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 1] = bbox[1] # cy
                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 2] = bbox[2] # o_w
                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 3] = bbox[3] # o_h
                        

                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 4] = object_score
                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 5] = lamda # lamda parameter
                    
                    else:# The point is not in positive anchor points.
                        class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + 0] = 0

        class_label_map = class_label_map.reshape(1, int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### 1,h,w,c
        points_label_map = points_label_map.reshape(1, int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### 1,h,w,c
        class_label_map = torch.from_numpy(class_label_map)
        points_label_map = torch.from_numpy(points_label_map)
        label_list.append(class_label_map)
        label_list.append(points_label_map)
        
        return label_list
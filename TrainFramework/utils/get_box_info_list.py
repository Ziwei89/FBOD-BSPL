import torch
import torch.nn as nn
import sys
import math
import numpy as np
import copy
sys.path.append("..")

    
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
        self.post_score = None

    @property
    def box_id(self):
        return self.__box_id
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
    
    def forward(self, input, bboxes):
       # input is a [CONF, LOC] list with 'bs,c,h,w' format tensor.
        bs = input[0].size(0)
        if bs > 1:
            raise print("Error! Can't process multi-batch!")

        # Branch for task, there are 2 tasks, that is CONF(Conf), and LOC(LOCation).
        ################# CONF #############################
        predict_CONF = input[0] ## 1,1,h,w
        predict_CONF = predict_CONF.squezze() ### h,w

        box_info_list = self.__get_boxes_info(bboxes=bboxes)
        for box_info in box_info_list:
            predict_conf = predict_CONF* box_info.positive_points_map
            predict_conf = torch.max(predict_conf)
            box_info.post_score = predict_conf
        return box_info_list
      
    def __get_boxes_info(self, bboxes): ###
        ###  bbox[0] x1, bbox[1] y1, bbox[2] x2, bbox[3] y2, bbox[4] class_id, bbox[5] object score (difficult) ###
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

        for box_id, bbox in enumerate(bboxes):
            obj_area = bbox[2] * bbox[3]
            if obj_area == 0:
                continue
            box_id_list.append(box_id)
            ###  bbox[0] cx, bbox[1] cy, bbox[2] o_w, bbox[3] o_h, bbox[4] class_id, bbox[5] difficult ###
            box_info_list.append(Box_info(box_id=box_id, bbox=bbox, xyxy=False))

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
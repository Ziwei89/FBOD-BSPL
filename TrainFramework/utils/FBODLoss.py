import torch
import torch.nn as nn
import sys
import math
import copy
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

class LossFunc(nn.Module): #
    def __init__(self,num_classes, model_input_size=(672,384), scale=80., m=1/3, stride=2, learn_mode="SPL", cuda=True, gettargets=False):
        super(LossFunc, self).__init__()
        self.num_classes = num_classes
        self.model_input_size = model_input_size
        self.scale = scale
        self.learn_mode = learn_mode
        #(model_input_size, num_classes=2, stride=2)
        self.get_targets = getTargets(model_input_size, num_classes, scale=scale, m=m, stride=stride, cuda=True)
        self.cuda = cuda
        self.gettargets = gettargets
    
    def forward(self, input, targets, spl_threshold=None):

        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        # targets is bboxes, bbox[0] cx, bbox[1] cy, bbox[2] w, bbox[3] h, bbox[4] class_id, bbox[5] score
        if self.gettargets:
            if self.learn_mode == "All_Sample":
                targets = self.get_targets(input, targets, difficult_mode=0) ### targets is a list wiht 2 members, each is a 'bs,in_h,in_w,c' format tensor(cls and bbox).
            elif self.learn_mode == "Easy_Sample":
                targets = self.get_targets(input, targets, difficult_mode=1) ### targets is a list wiht 2 members, each is a 'bs,in_h,in_w,c' format tensor(cls and bbox).
            elif self.learn_mode == "SPLBC":
                targets = self.get_targets(input, targets, difficult_mode=2, spl_threshold=spl_threshold)
            elif self.learn_mode == "SPL" or self.learn_mode == "HEM":
                targets = self.get_targets(input, targets, difficult_mode=3)
            else:
                raise("Error! learn_mode error.")

        # input is a list with with 2 members(CONF and LOC), each member is a 'bs,c,in_h,in_w' format tensor).
        # print("input[0].size()")
        # print(input[0].size())
        # print("input[1].size()")
        # print(input[1].size())
        bs = input[0].size(0)
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
        # print("label_CONF_CLS[:,:,:,1:].size()")
        # print(label_CONF_CLS[:,:,:,1:].size())
        label_CONF = torch.sum(label_CONF_CLS[:,:,:,1:], dim=3) # bs, in_h, in_w ## Guassian Heat Conf
        # print("label_CONF.size()")
        # print(label_CONF.size())

        label_CLS_weight =  torch.ceil(label_CONF_CLS) # bs,in_h,in_w,c
        weight_neg = label_CLS_weight[:,:,:,:1] # bs,in_h,in_w,c(c = 1)
        if self.num_classes > 2:
            weight_non_ignore = torch.sum(label_CLS_weight,3).unsqueeze(3)
            weight_pos = (1 - weight_neg)*weight_non_ignore # Exclude rows with all zeros.
        else:
            weight_pos = label_CLS_weight[:,:,:,1:] # bs,in_h,in_w,c(c = 1)

        ### bs,in_h,in_w
        weight_neg = weight_neg.squeeze(3)
        ### bs,in_h,in_w
        weight_pos = weight_pos.squeeze(3)
        ### bs
        bs_neg_nums = torch.sum(weight_neg, dim=(1,2))
        ### bs
        bs_obj_nums = torch.sum(weight_pos, dim=(1,2))
        
        #################LOC
        label_LOC_sampleweight_lamda = targets[1].type(FloatTensor) #bs*in_h,in_w*c  c=6(cx,xy,o_w,o_h,difficult,lamda)
        label_LOC_sampleweight_lamda = label_LOC_sampleweight_lamda.view(bs,in_h,in_w,-1) # bs,in_h,in_w,c(c=6)
        ### bs, in_h, in_w, c(c=4 cx,xy,o_w,o_h)
        label_LOC = label_LOC_sampleweight_lamda[:,:,:,:4] # bs,in_h,in_w,c(c=4)
        ### bs, in_h, in_w
        label_sampleweight = label_LOC_sampleweight_lamda[:,:,:,4] # bs,in_h,in_w
        ### bs, in_h, in_w
        # label_lamda = label_LOC_sampleweight_lamda[:,:,:,5] # bs,in_h,in_w

        ## Conf Loss
        ## bs, in_h, in_w
        # print("predict_CONF[predict_CONF>0.2]")
        # print(predict_CONF[predict_CONF>0.2])
        MSE_Loss = MSELoss(label_CONF, predict_CONF)
        neg_MSE_Loss = MSE_Loss * weight_neg
        pos_MSE_Loss = (MSE_Loss * label_sampleweight) * weight_pos

        CONF_loss = 0
        for b in range(bs):
            CONF_loss_per_batch = 0
            ### in_h, in_w
            if bs_obj_nums[b] != 0:
                k = bs_obj_nums[b].cpu()
                k = int(k.numpy())
                topk = 2*k
                if topk > bs_neg_nums[b]:
                    topk = bs_neg_nums[b]
                neg_MSE_Loss_topk_sum = torch.sum(torch.topk((neg_MSE_Loss[b]).view(-1), topk).values)
                pos_MSE_Loss_sum = torch.sum(pos_MSE_Loss[b])
                CONF_loss_per_batch = (neg_MSE_Loss_topk_sum + 10*pos_MSE_Loss_sum)/bs_obj_nums[b]
            else:
                neg_MSE_Loss_topk_sum = torch.sum(torch.topk(neg_MSE_Loss[b], 20).values)
                CONF_loss_per_batch = neg_MSE_Loss_topk_sum/10
            CONF_loss += CONF_loss_per_batch
        
        ### Locate Loss
        ciou_loss = 1-box_ciou(predict_LOC, label_LOC)
        ###(bs, in_h, in_w)
        ciou_loss = (ciou_loss.view(bs,in_h,in_w)) * label_sampleweight * weight_pos
        LOC_loss = 0
        for b in range(bs):
            LOC_loss_per_batch = 0
            if bs_obj_nums[b] != 0:
                LOC_loss_per_batch = torch.sum(ciou_loss[b])/bs_obj_nums[b]
            else:
                LOC_loss_per_batch = 0
            LOC_loss += LOC_loss_per_batch

        total_loss = (10*CONF_loss + 100*LOC_loss) / bs
        return total_loss

################################# For multi scale #################################################
def Distance(pointa,pointb):
    return math.sqrt((pointa[0]-pointb[0])**2+(pointa[1]-pointb[1])**2)

def filter_outermin_scale_targets(targets, bs, min_scale):
    ###Targets smaller than the minimum size are filtered out
    ### target [x1,y1,x2,y2]
    if min_scale==None:
        raise("Error! min_scale=None")
    new_targets = []
    for b in range(bs):
        target_list = []
        if len(targets[b]) == 0:
            new_targets.append(target_list)
            continue
        for target in targets[b]:
            if Distance([target[0],target[1]], [target[2],target[3]]) > min_scale:
                target = target.unsqueeze(0)
                target_list.append(target)
        if len(target_list) != 0:
            target_list = torch.concat(target_list,dim=0)
        new_targets.append(target_list)
    return new_targets

    

def filter_outermax_scale_targets(targets, bs, max_scale):
    ###Targets larger than the maximum size are filtered out
    ### target [x1,y1,x2,y2]
    if max_scale==None:
        raise("Error! max_scale=None")
    new_targets = []
    for b in range(bs):
        target_list = []
        if len(targets[b]) == 0:
            new_targets.append(target_list)
            continue
        for target in targets[b]:
            if Distance([target[0],target[1]], [target[2],target[3]]) <= max_scale:
                target = target.unsqueeze(0)
                target_list.append(target)
        if len(target_list) != 0:
            target_list = torch.concat(target_list,dim=0)
        new_targets.append(target_list)
    return new_targets

class LossFuncM(nn.Module): #
    def __init__(self,num_classes, model_input_size=(672,384), scale_list=[None, None], stride=2, cuda=True,gettargets=False):
        super(LossFuncM, self).__init__()
        #### scale_list: minimum size, maximum size (length)
        self.num_classes = num_classes
        self.model_input_size = model_input_size
        self.scale_list = scale_list
        #(model_input_size, num_classes=2, stride=2,8 or 32)
        self.get_targets = getTargets(model_input_size, num_classes, scale_list[1], stride, cuda) ### normalization scale use maximum(length)
        self.cuda = cuda
        self.gettargets = gettargets
    
    def forward(self, input, targets_all):
        # input is a list with with 2 members(CONF and LOC), each member is a 'bs,c,in_h,in_w' format tensor).
        # print("input[0].size()")
        # print(input[0].size())
        # print("input[1].size()")
        # print(input[1].size())
        bs = input[0].size(0)
        in_h = input[0].size(2) # in_h = model_input_size[1]/stride (stride = 2,8 or 32)
        in_w = input[0].size(3) # in_w

        targets = copy.deepcopy(targets_all)
        #### filter the outer scale targets
        targets = filter_outermin_scale_targets(targets, bs, self.scale_list[0])
        targets = filter_outermax_scale_targets(targets, bs, self.scale_list[1])

        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        # targets is bboxes, bbox[0] cx, bbox[1] cy, bbox[2] w, bbox[3] h, bbox[4] class_id, bbox[5] score
        if self.gettargets:
            targets = self.get_targets(input, targets) ### targets is a list wiht 2 members, each is a 'bs,in_h,in_w,c' format tensor(cls and bbox).

        

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
        
        predict_LOC[..., 0] = predict_LOC[..., 0]*self.scale_list[1] + ref_point_xs
        predict_LOC[..., 1] = predict_LOC[..., 1]*self.scale_list[1] + ref_point_ys
        predict_LOC[..., 2] = predict_LOC[..., 2]*self.scale_list[1] + ref_point_xs
        predict_LOC[..., 3] = predict_LOC[..., 3]*self.scale_list[1] + ref_point_ys

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
        # print("label_CONF_CLS[:,:,:,1:].size()")
        # print(label_CONF_CLS[:,:,:,1:].size())
        label_CONF = torch.sum(label_CONF_CLS[:,:,:,1:], dim=3) # bs, in_h, in_w ## Guassian Heat Conf
        # print("label_CONF.size()")
        # print(label_CONF.size())

        label_CLS_weight =  torch.ceil(label_CONF_CLS) # bs,in_h,in_w,c
        weight_neg = label_CLS_weight[:,:,:,:1] # bs,in_h,in_w,c(c = 1)
        if self.num_classes > 2:
            weight_non_ignore = torch.sum(label_CLS_weight,3).unsqueeze(3)
            weight_pos = (1 - weight_neg)*weight_non_ignore # Exclude rows with all zeros.
        else:
            weight_pos = label_CLS_weight[:,:,:,1:] # bs,in_h,in_w,c(c = 1)

        ### bs,in_h,in_w
        weight_neg = weight_neg.squeeze(3)
        ### bs,in_h,in_w
        weight_pos = weight_pos.squeeze(3)
        ### bs
        bs_neg_nums = torch.sum(weight_neg, dim=(1,2))
        ### bs
        bs_obj_nums = torch.sum(weight_pos, dim=(1,2))
        
        #################LOC
        label_LOC_sampleweight_lamda = targets[1].type(FloatTensor) #bs*in_h,in_w*c  c=6(cx,xy,o_w,o_h,difficult,lamda)
        label_LOC_sampleweight_lamda = label_LOC_sampleweight_lamda.view(bs,in_h,in_w,-1) # bs,in_h,in_w,c(c=6)
        ### bs, in_h, in_w, c(c=4 cx,xy,o_w,o_h)
        label_LOC = label_LOC_sampleweight_lamda[:,:,:,:4] # bs,in_h,in_w,c(c=4)
        ### bs, in_h, in_w
        # label_sampleweight = label_LOC_sampleweight_lamda[:,:,:,4] # bs,in_h,in_w
        ### bs, in_h, in_w
        # label_lamda = label_LOC_sampleweight_lamda[:,:,:,5] # bs,in_h,in_w

        ## Guassian Conf Loss
        ## bs, in_h, in_w
        # print("predict_CONF[predict_CONF>0.2]")
        # print(predict_CONF[predict_CONF>0.2])
        MSE_Loss = MSELoss(label_CONF, predict_CONF)
        neg_MSE_Loss = MSE_Loss * weight_neg
        pos_MSE_Loss = MSE_Loss * weight_pos

        CONF_loss = 0
        for b in range(bs):
            CONF_loss_per_batch = 0
            ### in_h, in_w
            if bs_obj_nums[b] != 0:
                k = bs_obj_nums[b].cpu()
                k = int(k.numpy())
                topk = 2*k
                if topk > bs_neg_nums[b]:
                    topk = bs_neg_nums[b]
                neg_MSE_Loss_topk_sum = torch.sum(torch.topk((neg_MSE_Loss[b]).view(-1), topk).values)
                pos_MSE_Loss_sum = torch.sum(pos_MSE_Loss[b])
                CONF_loss_per_batch = (neg_MSE_Loss_topk_sum + 10*pos_MSE_Loss_sum)/bs_obj_nums[b]
            else:
                neg_MSE_Loss_topk_sum = torch.sum(torch.topk(neg_MSE_Loss[b], 20).values)
                CONF_loss_per_batch = neg_MSE_Loss_topk_sum/10
            CONF_loss += CONF_loss_per_batch
        
        ### Locate Loss
        ciou_loss = 1-box_ciou(predict_LOC, label_LOC)
        ###(bs, in_h, in_w)
        ciou_loss = (ciou_loss.view(bs,in_h,in_w)) * weight_pos
        LOC_loss = 0
        for b in range(bs):
            LOC_loss_per_batch = 0
            if bs_obj_nums[b] != 0:
                LOC_loss_per_batch = torch.sum(ciou_loss[b])/bs_obj_nums[b]
            else:
                LOC_loss_per_batch = 0
            LOC_loss += LOC_loss_per_batch

        total_loss = (10*CONF_loss + 100*LOC_loss) / bs
        return total_loss
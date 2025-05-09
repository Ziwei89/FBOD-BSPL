from mAP import mean_average_precision
from FB_detector import FB_detector
import os
import cv2
import numpy as np
from PIL import Image
from queue import Queue
from utils.common import GetMiddleImg_ModelInput
import xml.etree.ElementTree as ET
from config.opts import opts
from utils.utils import FBObj

classes=['bird']
def ConvertAnnotationLabelToFBObj(annotation_file, image_id):
    label_obj_list = []
    in_file = open(annotation_file, encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
            
        cls = obj.find('name').text
        if cls not in classes:
            continue
        xmlbox = obj.find('bndbox')
        bbox = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)]
        # print("label:")
        # print(bbox)
        # print(cls_id)
        label_obj_list.append(FBObj(score=1.0, image_id=image_id, bbox=bbox))
    return label_obj_list

if __name__ == "__main__":
    opt = opts().parse()
    model_input_size = (int(opt.model_input_size.split("_")[0]), int(opt.model_input_size.split("_")[1])) # H,W

    input_img_num = opt.input_img_num
    aggregation_output_channels = opt.aggregation_output_channels
    aggregation_method = opt.aggregation_method
    input_mode=opt.input_mode
    backbone_name = opt.backbone_name
    fusion_method = opt.fusion_method
    learn_mode = opt.learn_mode
    # assign_method: The label assign method. binary_assign, guassian_assign or auto_assign
    if opt.assign_method == "binary_assign":
        abbr_assign_method = "ba"
    elif opt.assign_method == "guassian_assign":
        abbr_assign_method = "ga"
    elif opt.assign_method == "auto_assign":
        abbr_assign_method = "aa"
    else:
        raise("Error! abbr_assign_method error.")
    
    if opt.learn_mode == "SPL":
        Add_name = opt.spl_mode + "_" + opt.Add_name
    else:
        Add_name = opt.Add_name
    model_name=opt.model_name

    # FB_detector parameters
    # model_input_size=(384,672),
    # input_img_num=5, aggregation_output_channels=16, aggregation_method="multiinput", input_mode="GRG", backbone_name="cspdarknet53",
    # Add_name="as_1021_1", model_name="FB_object_detect_model.pth",
    # scale=80.

    fb_detector = FB_detector(model_input_size=model_input_size,
                              input_img_num=input_img_num, aggregation_output_channels=aggregation_output_channels,
                              aggregation_method=aggregation_method, input_mode=input_mode, backbone_name=backbone_name, fusion_method=fusion_method,
                              learn_mode=learn_mode, abbr_assign_method=abbr_assign_method, Add_name=Add_name, model_name=model_name)


    label_path = opt.data_root_path + "labels/val/" #.xlm label file path

    video_path = opt.data_root_path + "videos/val/"

    continus_num = input_img_num
    
    image_total_id = 0
    all_label_obj_list = []
    all_obj_result_list = []
    video_names = os.listdir(video_path)
    label_name_list=os.listdir(label_path)

    for video_name in video_names:
        image_q = Queue(maxsize=continus_num)
        start_image_total_id = image_total_id

        cap=cv2.VideoCapture(video_path + video_name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ################# frames padding ################
        for i in range(int(continus_num/2)):
            black_image = np.zeros((height, width, 3), dtype=np.uint8)
            black_image = Image.fromarray(cv2.cvtColor(black_image,cv2.COLOR_BGR2RGB))
            image_q.put(black_image)
        #################################################

        frame_id = 0
        while (True):
            ret,frame=cap.read()
            if ret != True:
                break
            else:
                image_total_id += 1
                frame_id += 1
                image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                image_q.put(image)
                image_shape = np.array(np.shape(image)[0:2]) # image size is 1280,720; image array's shape is 720,1280

                frame_id_str = "%06d" % int((frame_id-1))
                label_name = video_name.split(".")[0] + "_" + frame_id_str + ".xml"
                all_label_obj_list += ConvertAnnotationLabelToFBObj(label_path + label_name, start_image_total_id + frame_id)

                if frame_id >= int(continus_num/2) + 1 and frame_id <= frame_count:

                    _, model_input = GetMiddleImg_ModelInput(image_q, model_input_size=model_input_size, continus_num=continus_num, input_mode=input_mode)
                    _ = image_q.get()
                    outputs = fb_detector.detect_image(model_input, raw_image_shape=image_shape)

                    obj_result_list = []
                    for output in outputs[0]: ###
                        # print(output)
                        box = [0,0,0,0]
                        box[0] = output[0].item()
                        box[1] = output[1].item()
                        box[2] = output[2].item()
                        box[3] = output[3].item()
                        # print("predict:")
                        # print(box)
                        score = output[4].item()
                        ### The output of detector is start from continus_num-int(continus_num/2) frame.
                        obj_result_list.append(FBObj(score=score, image_id=start_image_total_id + (frame_id-int(continus_num/2)), bbox=box))
                    all_obj_result_list += obj_result_list
                if frame_id == frame_count: ## Output the detection results of the last int(continus_num/2) frames of the video.
                    for n in range(1, int(continus_num/2)+1):
                        
                        black_image = np.zeros((height, width, 3), dtype=np.uint8)
                        black_image = Image.fromarray(cv2.cvtColor(black_image,cv2.COLOR_BGR2RGB))
                        image_q.put(black_image)
                        
                        _, model_input = GetMiddleImg_ModelInput(image_q, model_input_size=model_input_size, continus_num=continus_num, input_mode=input_mode)
                        _ = image_q.get()
                        outputs = fb_detector.detect_image(model_input, raw_image_shape=image_shape)

                        obj_result_list = []
                        for output in outputs[0]: ###
                            # print(output)
                            box = [0,0,0,0]
                            box[0] = output[0].item()
                            box[1] = output[1].item()
                            box[2] = output[2].item()
                            box[3] = output[3].item()
                            # print("predict:")
                            # print(box)
                            score = output[4].item()
                            ### The output of detector is delayed by int(continus_num/2) frames.
                            obj_result_list.append(FBObj(score=score, image_id=start_image_total_id + (frame_id-(int(continus_num/2)-n)), bbox=box))
                        all_obj_result_list += obj_result_list
    AP_50,REC_50,PRE_50=mean_average_precision(all_obj_result_list,all_label_obj_list,iou_threshold=0.5)
    print("AP_50,REC_50,PRE_50:")
    print(AP_50,REC_50,PRE_50)
    AP_75,REC_75,PRE_75=mean_average_precision(all_obj_result_list,all_label_obj_list,iou_threshold=0.75)
    print("AP_75,REC_75,PRE_75:")
    print(AP_75,REC_75,PRE_75)
    mAP = 0
    for i in range(50,95,5):
        iou_t = i/100
        mAP_, _, _ = mean_average_precision(all_obj_result_list,all_label_obj_list,iou_threshold=iou_t)
        mAP += mAP_
    mAP = mAP/10
    print("mAP = ",mAP)
from utils.get_box_info_list import image_info, getBoxInfoListForOneImage, object_score_to_HElevel
import os
from FB_detector import FB_detector
from utils.common import load_data, GetMiddleImg_ModelInput_for_MatImageList
from config.opts import opts
import glob

os.environ['KMP_DUPLICATE_LIB_OK']='True'

num_to_english_c_dic = {3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}

if __name__ == "__main__":
    opt = opts().parse()
    model_input_size = (int(opt.model_input_size.split("_")[0]), int(opt.model_input_size.split("_")[1])) # H,W

    input_img_num = opt.input_img_num
    aggregation_output_channels = opt.aggregation_output_channels
    aggregation_method = opt.aggregation_method
    input_mode=opt.input_mode
    backbone_name = opt.backbone_name
    fusion_method = opt.fusion_method
    # assign_method: The label assign method. binary_assign, guassian_assign or auto_assign
    if opt.assign_method == "binary_assign":
        abbr_assign_method = "ba"
    elif opt.assign_method == "guassian_assign":
        abbr_assign_method = "ga"
    elif opt.assign_method == "auto_assign":
        abbr_assign_method = "aa"
    else:
        raise("Error! abbr_assign_method error.")
    
    Add_name=opt.Add_name

    save_model_dir = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                             + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + abbr_assign_method + "_"  + opt.Add_name + "/"
    model_path = glob.glob(save_model_dir + "Epoch"+str(opt.end_epoch)+"_*")[0] ### Has wildcard in model name eg. Epoch20_*
    model_name=model_path.split("/")[-1] ### No wildcard in model name.

    # FB_detector parameters
    # model_input_size=(384,672),
    # input_img_num=5, aggregation_output_channels=16, aggregation_method="multiinput", input_mode="GRG", backbone_name="cspdarknet53",
    # Add_name="as_1021_1", model_name="FB_object_detect_model.pth",
    # scale=80.
    
    fb_detector = FB_detector(model_input_size=model_input_size,
                              input_img_num=input_img_num, aggregation_output_channels=aggregation_output_channels,
                              aggregation_method=aggregation_method, input_mode=input_mode, backbone_name=backbone_name, fusion_method=fusion_method,
                              abbr_assign_method=abbr_assign_method, Add_name=Add_name, model_name=model_name)
    Cuda = True
    annotation_path = "./dataloader/" + "img_label_" + num_to_english_c_dic[input_img_num] + "_continuous_difficulty_val.txt"
    dataset_image_path = opt.data_image_path + "val/"

    
    get_box_info_for_one_image = getBoxInfoListForOneImage(image_size = (model_input_size[1],model_input_size[0])) # image_size w,h
    
    image_info_list = []
    Count_HElevel = [0] * 4
    with open(annotation_path) as f:
        lines = f.readlines()
    for line in lines:
        images, bboxes, first_img_name = load_data(line, dataset_image_path, frame_num=input_img_num, image_size=model_input_size, needToresize=True)
        image_info_instance = image_info(iname=first_img_name)
        if len(bboxes) == 0:
            image_info_list.append(image_info_instance)
            continue

        for bbox in bboxes:
            Count_HElevel[object_score_to_HElevel(bbox[5])] += 1
        _, model_input= GetMiddleImg_ModelInput_for_MatImageList(images, model_input_size=model_input_size, continus_num=input_img_num, input_mode=input_mode)
        predictions = fb_detector.inference(model_input)
        image_info_instance.box_info_list = get_box_info_for_one_image(predictions, bboxes)
        image_info_list.append(image_info_instance)
    
    
    P_Upper_Lower=[[0,0]]*4 ### The Upper and Lower value of the Post confidence for four HElevels
    S_Upper_Lower = [[1, 0.75],[0.75, 0.5],[0.5, 0.25],[0.25, 0]] ### The Upper and Lower object score for four HElevels

    post_conf_list = []
    for image_info_instance in image_info_list:
        for box_info_instance in image_info_instance.box_info_list:
            post_conf_list.append(box_info_instance.post_score)
    post_conf_list.sort(reverse=True)

    for i in range(4):
        if i == 0 :
            P_Upper_Lower[i][0] = post_conf_list[0]
        else:
            P_Upper_Lower[i][0] = post_conf_list[Count_HElevel[i-1] + 1]
        P_Upper_Lower[i][1] = post_conf_list[Count_HElevel[i]]
    
    P_Max_Min=[[0,0]]*4 ### The Max and Min value of the Post confidence for four HElevels
    S_Max_Min = [[0,0]]*4 ### The Max and Min object score for four HElevels
    #### To determin the P_Max_Min and S_Max_Min through regulation
    for i in range(4):
        if (P_Upper_Lower[i][0] <= S_Upper_Lower[i][0]) and (P_Upper_Lower[i][1] <= S_Upper_Lower[i][1]):
            P_Max_Min[i][0] = S_Upper_Lower[i][0] ### Pmax
            S_Max_Min[i][0] = S_Upper_Lower[i][0] ### Smax

            P_Max_Min[i][1] = P_Upper_Lower[i][1] ### Pmin
            S_Max_Min[i][1] = S_Upper_Lower[i][1] ### Smin

        elif (P_Upper_Lower[i][0] <= S_Upper_Lower[i][0]) and (P_Upper_Lower[i][1] >= S_Upper_Lower[i][1]):
            P_Max_Min[i][0] = S_Upper_Lower[i][0] ### Pmax
            S_Max_Min[i][0] = S_Upper_Lower[i][0] ### Smax

            P_Max_Min[i][1] = P_Upper_Lower[i][1] ### Pmin
            S_Max_Min[i][1] = S_Upper_Lower[i][1] ### Smin

        elif (P_Upper_Lower[i][0] >= S_Upper_Lower[i][0]) and (P_Upper_Lower[i][1] <= S_Upper_Lower[i][1]):
            P_Max_Min[i][0] = P_Upper_Lower[i][0] ### Pmax
            S_Max_Min[i][0] = S_Upper_Lower[i][0] ### Smax

            P_Max_Min[i][1] = P_Upper_Lower[i][1] ### Pmin
            S_Max_Min[i][1] = S_Upper_Lower[i][1] ### Smin
        
        elif (P_Upper_Lower[i][0] >= S_Upper_Lower[i][0]) and (P_Upper_Lower[i][1] >= S_Upper_Lower[i][1]):
            P_Max_Min[i][0] = P_Upper_Lower[i][0] ### Pmax
            S_Max_Min[i][0] = S_Upper_Lower[i][0] ### Smax

            P_Max_Min[i][1] = S_Upper_Lower[i][1] ### Pmin
            S_Max_Min[i][1] = S_Upper_Lower[i][1] ### Smin
    
    ### 
    for image_info_instance in image_info_list:
        for box_info_instance in image_info_instance.box_info_list:
            p = box_info_instance.post_score
            if p > P_Upper_Lower[0][1]:
                box_info_instance.post_HElevel = 0
            elif p > P_Upper_Lower[1][1]:
                box_info_instance.post_HElevel = 1
            elif p > P_Upper_Lower[2][1]:
                box_info_instance.post_HElevel = 2
            elif p > P_Upper_Lower[3][1]:
                box_info_instance.post_HElevel = 3
            
            index = box_info_instance.post_HElevel
            p_max = P_Max_Min[index][0]
            p_min = P_Max_Min[index][1]
            s_max = S_Max_Min[index][0]
            s_min = S_Max_Min[index][1]

            box_info_instance.prior_score = (s_max*(p-p_min) + s_min*(p_max-p))/(p_max-p_min)
    
    ### Update the object score of annotation by rewriting all the information.
    annotation_file = open(annotation_path,'w')
    for image_info_instance in image_info_list:
        annotation_file.write(image_info_instance.iname)
        if len(image_info_instance.box_info_list) == 0:
            annotation_file.write(" None")
        else:
            for box_info_instance in image_info_instance.box_info_list:
                string_label = " " + ",".join(str(a) for a in box_info_instance.bbox) + "," + str(int(box_info_instance.class_id)) + "," + str(box_info_instance.prior_score)
                annotation_file.write(string_label)
        annotation_file.write("\n")
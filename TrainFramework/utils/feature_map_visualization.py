import cv2
import numpy as np
import copy
import torch

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
 
def save_grayscale_3D(gray_image, pic_save_dir, img_name):
    fig = plt.figure()
    ax = fig.add_axes(Axes3D(fig))
    
    # Make data.
    height, width = gray_image.shape
    X = np.arange(0, width, 1)
    Y = np.arange(0, height, 1)
    X, Y = np.meshgrid(X, Y)
    
    Z = np.array(gray_image)
    
    # Plot the surface.
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis_r')
    ax.plot_surface(X, Y, Z, cmap='jet',linewidth=0, antialiased=False)
    
    # Customize the z axis.
    ax.set_zlim(0, 1.0)
    ax.invert_yaxis()
    ax.zaxis.set_major_locator(LinearLocator(8))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig(pic_save_dir + img_name,bbox_inches='tight')

def create_grayscale_heatmap(image):
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return heatmap

def show_feature_map(in_feature_map, img_name="feature_map.jpg", model_input_size=(672,384), input_img_size=(1280,720), pic_save_dir="./test_output/"):
    resize_ratio = min(1.0 * model_input_size[0] / input_img_size[0], 1.0 * input_img_size[1] / input_img_size[1])
    resize_w = int(resize_ratio * input_img_size[0])
    resize_h = int(resize_ratio * input_img_size[1])

    dh = model_input_size[1] - resize_h
    top = int(dh/2)
    bottom = dh - top

    dw = model_input_size[0] - resize_w
    left = int(dw/2)
    right = dw - left

    crop_box = [left, top, model_input_size[0]-right, model_input_size[1]-bottom] #x1,y1, x2,y2

    feature_map_data_list = [in_feature_map[0, i].cpu().detach().numpy() for i in range(in_feature_map.shape[1])]
    feature_map = np.zeros(shape=(input_img_size[1], input_img_size[0]))
    for feature_map_data in feature_map_data_list:
        feature_map_data = cv2.resize(feature_map_data, model_input_size)
        feature_map_data_crop = copy.deepcopy(feature_map_data[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]])
        feature_map_data = cv2.resize(feature_map_data_crop, input_img_size)
        feature_map += feature_map_data
    pmin = np.min(feature_map)
    pmax = np.max(feature_map)
    feature_map = (feature_map - pmin) / (pmax - pmin + 0.000001)
    
    save_grayscale_3D(feature_map, pic_save_dir, "3d_" + img_name)
    # feature_map = np.asarray(feature_map * 255, dtype=np.uint8)
    # heatmap = create_grayscale_heatmap(feature_map)
    # prefix_name = img_name.split(".")[0]
    # prefix_name_list = prefix_name.split("_")
    # new_num_str = "%06d" % (int(prefix_name_list[2]) + 2)
    # img_name = prefix_name_list[0] + "_" + prefix_name_list[1] + "_" + new_num_str + ".jpg"
    # cv2.imwrite(pic_save_dir + img_name, heatmap)


def show_out_feature_map(in_feature_map, img_name="feature_map.jpg", model_input_size=(672,384), input_img_size=(1280,720), pic_save_dir="./test_output/"):
    resize_ratio = min(1.0 * model_input_size[0] / input_img_size[0], 1.0 * input_img_size[1] / input_img_size[1])
    resize_w = int(resize_ratio * input_img_size[0])
    resize_h = int(resize_ratio * input_img_size[1])

    dh = model_input_size[1] - resize_h
    top = int(dh/2)
    bottom = dh - top

    dw = model_input_size[0] - resize_w
    left = int(dw/2)
    right = dw - left

    crop_box = [left, top, model_input_size[0]-right, model_input_size[1]-bottom] #x1,y1, x2,y2

    in_h = in_feature_map.size(2) # in_h = model_input_size[0]/2
    in_w = in_feature_map.size(3) # in_w

    in_feature_map = in_feature_map.view(in_h, in_w) #bs,h,w
    in_feature_map = torch.sigmoid(in_feature_map)
    feature_map = in_feature_map.cpu().detach().numpy()

    feature_map = cv2.resize(feature_map, model_input_size)
    feature_map = copy.deepcopy(feature_map[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]])
    feature_map = cv2.resize(feature_map, input_img_size)
    
    save_grayscale_3D(feature_map, pic_save_dir, "3d_" + img_name)
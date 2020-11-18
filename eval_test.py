import os
import torch
import torchvision.models as models
import cv2
#import cPickle
import numpy as np
import importlib
import json

from model_defs.TDID import TDID
from model_defs.nms.nms_wrapper import nms
from utils import * 

import active_vision_dataset_processing.data_loading.active_vision_dataset as AVD  


def im_detect(net, target_data,im_data, im_info, features_given=True):
    """
    Detect single target object in a single scene image.

    Input Parameters:
        net: (TDID) the network
        target_data: (torch Variable) target images
        im_data: (torch Variable) scene_image
        im_info: (tuple) (height,width,channels) of im_data
        
        features_given(optional): (bool) if true, target_data and im_data
                                  are feature maps from net.features,
                                  not images. Default: True
                                    

    Returns:
        scores (ndarray): N x 2 array of class scores
                          (N boxes, classes={background,target})
        boxes (ndarray): N x 4 array of predicted bounding boxes
    """

    cls_prob, rois = net(target_data, im_data, im_info,
                                    features_given=features_given)
    scores = cls_prob.data.cpu().numpy()[0,:,:]
    zs = np.zeros((scores.size, 1))
    scores = np.concatenate((zs,scores),1)
    boxes = rois.data.cpu().numpy()[0,:, :]

    return scores, boxes

if __name__ == '__main__':

    #load config file
    cfg_file = 'configAVD1' #NO EXTENSTION!
    cfg = importlib.import_module('configs.'+cfg_file)
    cfg = cfg.get_config()

    # load net
    print('Loading ' + cfg.FULL_MODEL_LOAD_NAME + ' ...')
    net = TDID(cfg)
    load_net(cfg.FULL_MODEL_LOAD_DIR + cfg.FULL_MODEL_LOAD_NAME, net)
    net.features.eval()#freeze batchnorms layers?
    print('load model successfully!')
    
    net.cuda()
    net.eval()

    print("Done")





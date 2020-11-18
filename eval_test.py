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

import random
def load_image():

  pathToFolder = '/content/drive/My Drive/Data/SyntheticDataset/'

  f=open(pathToFolder + "info.csv")
  lines=f.readlines()

  data = random.choice(lines).split("\t")
  image_name = data[0]
  target_name_1 = data[5]
  target_name_2 = data[6]

  pre_load_image = cv2.imread(os.path.join(pathToFolder, "Data", image_name))
  pre_load_target_1 = cv2.imread(os.path.join(pathToFolder, "Data", target_name_1))
  pre_load_target_2 = cv2.imread(os.path.join(pathToFolder, "Data", target_name_2))

  image = cv2.resize(pre_load_image, (1920, 1080), interpolation = cv2.INTER_AREA)
  bbox = [int(data[1]), int(data[2]), int(data[3]), int(data[4]), 1]
  target1 = cv2.resize(pre_load_target_1, (80, 80), interpolation = cv2.INTER_AREA)
  target2 = cv2.resize(pre_load_target_2, (80, 80), interpolation = cv2.INTER_AREA)

  return image, bbox, target1, target2

if __name__ == '__main__':

  #load config file
  cfg_file = 'configAVD1' #NO EXTENSTION!
  cfg = importlib.import_module('configs.'+cfg_file)
  cfg = cfg.get_config()

  print('Loading network...')
  net = TDID(cfg)
  if cfg.LOAD_FULL_MODEL:
      load_net(cfg.FULL_MODEL_LOAD_DIR + cfg.FULL_MODEL_LOAD_NAME, net)
  else:
      weights_normal_init(net, dev=0.01)
      if cfg.USE_PRETRAINED_WEIGHTS:
          net.features = load_pretrained_weights(cfg.FEATURE_NET_NAME) 
  net.features.eval()#freeze batchnorms layers?

  if not os.path.exists(cfg.SNAPSHOT_SAVE_DIR):
      os.makedirs(cfg.SNAPSHOT_SAVE_DIR)
  if not os.path.exists(cfg.META_SAVE_DIR):
      os.makedirs(cfg.META_SAVE_DIR)

  print('load model successfully!')

  net.cuda()
  net.eval()

  print("Done")





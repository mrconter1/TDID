import torch
import torch.utils.data
import torchvision.models as models
import os
import sys
import importlib
import numpy as np
from datetime import datetime
import cv2
import time

from model_defs.TDID import TDID 
from utils import *
from evaluation.coco_det_eval import coco_det_eval 

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

def find_files(path, file_ending):
  found_files = []
  for currentpath, folders, files in os.walk(path):
    for filename in files:
        filepath = os.path.join(currentpath, filename)
        if (filepath.endswith(file_ending)):
          found_files.append(filepath)
  return found_files

import random
def load_image():

  pathToBackgrounds = '/content/drive/My Drive/ActiveVisionDataset/'
  pathToGT = '/content/drive/My Drive/Data/GT/'
  pathToTargets = '/content/drive/My Drive/Data/AVD_and_BigBIRD_targets_v1/'

  valid_files = find_files(pathToBackgrounds, ".jpg")

  while True:

    chosen_image_path = random.choice(valid_files)
    chosen_image = chosen_image_path.split("/")[-1]

    json_files = find_files(pathToGT, ".json")
    json_data = ""
    for json_file in json_files:
      with open(json_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
          json_data += line

    if chosen_image in json_data:

      try:

        image_id = json_data.split(chosen_image)[1].split("}")[0]
        image_id = image_id.split("id\": ")[1].split(",")[0]

        data = json_data.split("\"image_id\": " + image_id)[1].split("}")[0]

        bb_data = []
        bounding_boxes_data = data.split("\"bbox\": [")[1].split("]")[0]
        for value in bounding_boxes_data.split(","):
          bb_data.append(int(value))

        category_id = int(data.split("\"category_id\": ")[1].split(",")[0])

        target_name = ""
        with open(pathToBackgrounds + "instance_id_map.txt", 'r') as file:
          lines = file.readlines()
          for line in lines:
            line = line.rstrip()
            linesplit = line.split(" ")
            name = linesplit[0]
            cat_num = int(linesplit[1])
            if cat_num == category_id:
              target_name = name

        target_paths = find_files(pathToTargets, ".jpg")
        target_image_paths_1 = []
        target_image_paths_2 = []
        for target_path in target_paths:
          if target_name in target_path:
            if "target_0" in target_path:
              target_image_paths_1.append(target_path) 
            elif "target_1" in target_path:
              target_image_paths_2.append(target_path) 
        
        pre_load_image = cv2.imread(chosen_image_path)
        pre_load_target_1 = cv2.imread(random.choice(target_image_paths_1))
        pre_load_target_2 = cv2.imread(random.choice(target_image_paths_2))

        
        image = cv2.resize(pre_load_image, (1920, 1080), interpolation = cv2.INTER_AREA)
        bbox = [bb_data[0], bb_data[1], bb_data[2], bb_data[3], 1]
        target1 = cv2.resize(pre_load_target_1, (80, 80), interpolation = cv2.INTER_AREA)
        target2 = cv2.resize(pre_load_target_2, (80, 80), interpolation = cv2.INTER_AREA)

        return image, bbox, target1, target2
      
      except Exception as e:
        print(e)

# load config
cfg_file = "configAVD1"
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

#put net on gpu
net.cuda()
net.eval()

batch_im_data = []
batch_target_data = []
batch_gt_boxes = []

im_data, gt_boxes, target1, target2 = load_image()

batch_im_data.append(normalize_image(im_data,cfg))
batch_gt_boxes.extend(gt_boxes)
batch_target_data.append(normalize_image(target1,cfg))
batch_target_data.append(normalize_image(target2,cfg))

#prep data for input to network
target_data = match_and_concat_images_list(batch_target_data,
                                            min_size=cfg.MIN_TARGET_SIZE)
im_data = match_and_concat_images_list(batch_im_data)
gt_boxes = np.asarray(batch_gt_boxes) 
im_info = im_data.shape[1:]
im_data = np_to_variable(im_data, is_cuda=True)
im_data = im_data.permute(0, 3, 1, 2).contiguous()
target_data = np_to_variable(target_data, is_cuda=True)
target_data = target_data.permute(0, 3, 1, 2).contiguous()

scores, boxes = im_detect(net, target_data, im_data, im_info, features_given=False)

print(boxes[0])

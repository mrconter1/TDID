import time
import numpy as np
import importlib
from PIL import Image
from model_defs.nms.nms_wrapper import nms
import random
from utils import *
from evaluation.coco_det_eval import coco_det_eval 
from model_defs.TDID import TDID

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import subprocess

import active_vision_dataset_processing.data_loading.active_vision_dataset as AVD 

def strToNum(s):
  hashed = abs(hash(s)) % (1000)
  return hashed

fileName = str(random.randint(0, 1000000))
print("Filename: " + fileName)

def im_detect(net, target_data,im_data, im_info, features_given=False):
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

def writeForPASCALVOC(path, filename, catName, data):
  outStr = catName + " "
  for i in range(len(data)):
    outStr += str(float(data[i])) + " "
  with open(path + filename + ".txt", 'a') as out:
      out.write(outStr + '\n') 

def load_synth_image(training):

  pathToFolder = '/content/drive/My Drive/Data/SyntheticDataset_new/'

  f=open(pathToFolder + "info.csv")

  lines=f.readlines()
  trainRatio = 0.6
  if train:
    lines=lines[:int(len(lines)*trainRatio)]
  else:
    lines=lines[int(len(lines)*trainRatio):]

  Fail = True
  while Fail:

    try:
      data = random.choice(lines).split("\t")
      image_name = data[0]
      target_name_1 = data[5]
      target_name_2 = data[6]

      image_path = os.path.join(pathToFolder, "Data", image_name)
      pre_load_image = cv2.imread(image_path)
      pre_load_target_1 = cv2.imread(os.path.join(pathToFolder, "Data", target_name_1))
      pre_load_target_2 = cv2.imread(os.path.join(pathToFolder, "Data", target_name_2))

      image = cv2.resize(pre_load_image, (1920, 1080), interpolation = cv2.INTER_AREA)
      bbox = [int(data[1]), int(data[2]), int(data[3]), int(data[4]), 1]
      target1 = cv2.resize(pre_load_target_1, (80, 80), interpolation = cv2.INTER_AREA)
      target2 = cv2.resize(pre_load_target_2, (80, 80), interpolation = cv2.INTER_AREA)
      Fail = False
    except:
      Fail = True

  target = target_name_1.split("0")[0]
  category_id = strToNum(target)

  return image, bbox, target1, target2, image_path, category_id

def load_image(valid_files, training):

  ratio = 0.8
  if training:
    files = valid_files[:int(len(valid_files)*ratio)]
  else:
    files = valid_files[int(len(valid_files)*ratio):]

  image, bbox, target1, target2, image_path, category_id = None, None, None, None, None, None

  while True:

    try:

      chosen_image_path = random.choice(files)
      chosen_image = chosen_image_path.split("/")[-1]

      json_files = find_files(pathToGT, ".json")
      json_data = ""
      for json_file in json_files:
        with open(json_file, 'r') as file:
          lines = file.readlines()
          for line in lines:
            json_data += line

      if chosen_image in json_data:

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
              break

        batch_im_data = []
        batch_target_data = []
        batch_gt_boxes = []

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

        break

    except:

      continue

  target = target_image_paths_1[0].split('/')[-1].split('.')[0][:-2]
  category_id = strToNum(target)
  
  return image, bbox, target1, target2, chosen_image_path, category_id

###
import subprocess
import time

loadNet = False
train = True
data_type = 'synthetic' #'synthetic' or 'AVD'

print("Config")
cfg_file = "configAVD1"
cfg = importlib.import_module('configs.'+cfg_file)
cfg = cfg.get_config()

print("Init net")
net = TDID(cfg)
print("Loading net")
if loadNet:
  load_net(cfg.FULL_MODEL_LOAD_DIR + cfg.FULL_MODEL_LOAD_NAME, net)
else:
  weights_normal_init(net, dev=0.01)
  net.features = load_pretrained_weights(cfg.FEATURE_NET_NAME) 
print("Freeze batchnorms")
net.features.eval()#freeze batchnorms layers?
print("cuda")
net.cuda()

pathToBackgrounds = '/content/drive/My Drive/ActiveVisionDataset/'
pathToGT = '/content/drive/My Drive/Data/GT/'
pathToTargets = '/content/drive/My Drive/Data/AVD_and_BigBIRD_targets_v1/'
valid_files = find_files(pathToBackgrounds, ".jpg")

if train:

  net.train()

  #Train
  numToTrainOn = 100000
  updateInterval = 250
  numToAvg = 50
  batchSize = 2

  params = list(net.parameters())

  optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.005)
  #optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, nesterov=True)
  #optimizer = torch.optim.Adam(params, lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
  #optimizer = torch.optim.RMSprop(params, lr=0.0001)

  trainLossListX = []
  trainLossListY = []

  valLossListX = []
  valLossListY = []

  accListX = []
  accListY = []

  mAP = 0

  for x in range(0, numToTrainOn):

    net.train()

    batch_im_data = []
    batch_target_data = []
    batch_gt_boxes = []

    for j in range(batchSize):

      if data_type == 'AVD':
        im_data, gt_boxes, target1, target2, image_path, category_id = load_image(valid_files, training = True)
      else:
        im_data, gt_boxes, target1, target2, image_path, category_id = load_synth_image(training = True)
    
      target1 = augment_image(target1, do_illum=cfg.AUGMENT_TARGET_ILLUMINATION)
      target2 = augment_image(target2, do_illum=cfg.AUGMENT_TARGET_ILLUMINATION)

      gt_boxes[2] += gt_boxes[0]
      gt_boxes[3] += gt_boxes[1]
      
      gt_boxes = np.asarray([[gt_boxes[0], gt_boxes[1], gt_boxes[2], gt_boxes[3], str(category_id)]],dtype=np.float32)

      batch_im_data.append(normalize_image(im_data,cfg))
      batch_gt_boxes.extend(gt_boxes)
      batch_target_data.append(normalize_image(target1,cfg))
      batch_target_data.append(normalize_image(target2,cfg))

    target_data = match_and_concat_images_list(batch_target_data,
                                                min_size=cfg.MIN_TARGET_SIZE)
    im_data = match_and_concat_images_list(batch_im_data)
    gt_boxes = np.asarray(batch_gt_boxes)
    im_info = im_data.shape[1:]
    im_data = np_to_variable(im_data, is_cuda=True)
    im_data = im_data.permute(0, 3, 1, 2)
    target_data = np_to_variable(target_data, is_cuda=True)
    target_data = target_data.permute(0, 3, 1, 2)

    net(target_data, im_data, im_info, gt_boxes=gt_boxes)
    loss = net.loss

    trainLossListX.append(x)
    trainLossListY.append(loss.data[0]) 

    optimizer.zero_grad()
    loss.backward()
    clip_gradient(net, 10.)
    optimizer.step()

    if x%updateInterval == 0:

      net.eval()

      valList = []
      for v in range(numToAvg):

        batch_im_data = []
        batch_target_data = []
        batch_gt_boxes = []

        for j in range(batchSize):

          if data_type == 'AVD':
            im_data, gt_boxes, target1, target2, image_path, category_id = load_image(valid_files, training = False)
          else:
            im_data, gt_boxes, target1, target2, image_path, category_id = load_synth_image(training = False)

          target1 = augment_image(target1, do_illum=cfg.AUGMENT_TARGET_ILLUMINATION)
          target2 = augment_image(target2, do_illum=cfg.AUGMENT_TARGET_ILLUMINATION)

          gt_boxes[2] += gt_boxes[0]
          gt_boxes[3] += gt_boxes[1]

          gt_boxes = np.asarray([[gt_boxes[0],gt_boxes[1],gt_boxes[2],gt_boxes[3],str(category_id)]])

          batch_im_data.append(normalize_image(im_data,cfg))
          batch_gt_boxes.extend(gt_boxes)
          batch_target_data.append(normalize_image(target1,cfg))
          batch_target_data.append(normalize_image(target2,cfg))

        target_data = match_and_concat_images_list(batch_target_data,
                                                    min_size=cfg.MIN_TARGET_SIZE)
        im_data = match_and_concat_images_list(batch_im_data)
        gt_boxes = np.asarray(batch_gt_boxes)
        im_info = im_data.shape[1:]
        im_data = np_to_variable(im_data, is_cuda=True)
        im_data = im_data.permute(0, 3, 1, 2)
        target_data = np_to_variable(target_data, is_cuda=True)
        target_data = target_data.permute(0, 3, 1, 2)

        net(target_data, im_data, im_info, gt_boxes=gt_boxes)
        loss = net.loss

        valList.append(loss.data[0]) 

      #mAP
      for m in range(numToAvg):

        batch_im_data = []
        batch_target_data = []
        batch_gt_boxes = []

        if data_type == 'AVD':
          im_data, gt_boxes, target1, target2, image_path, category_id = load_image(valid_files, training = False)
        else:
          im_data, gt_boxes, target1, target2, image_path, category_id = load_synth_image(training = False)

        batch_im_data.append(normalize_image(im_data,cfg))
        batch_gt_boxes.extend(gt_boxes)
        batch_target_data.append(normalize_image(target1,cfg))
        batch_target_data.append(normalize_image(target2,cfg))

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

        inds = np.where(scores[:, 1] > 0.1)[0]
        fg_scores = scores[inds, 1]
        fg_boxes = boxes[inds,:]
        fg_dets = np.hstack((fg_boxes, fg_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        keep = nms(fg_dets, 0.7)
        fg_dets = fg_dets[keep, :]

        gt_boxes[2] += gt_boxes[0]
        gt_boxes[3] += gt_boxes[1]

        for f in range(len(fg_dets)):

          detection = [fg_dets[f][4], fg_dets[f][0], fg_dets[f][1], fg_dets[f][2], fg_dets[f][3]]

          writeForPASCALVOC("Object-Detection-Metrics/detections/", fileName, str(category_id), detection)
          writeForPASCALVOC("Object-Detection-Metrics/groundtruths/", fileName, str(category_id), gt_boxes)

          if x > 5:
            break
          
      try:
        data = subprocess.check_output('python3 Object-Detection-Metrics/pascalvoc.py -t 0.5', shell=True)
        mAP = float(data.split('\n')[-2].split(' ')[1][:-2])
        accListX.append(x)
        accListY.append(mAP)
      except:
        accListX.append(x)
        accListY.append(0)

      valLossListX.append(x)
      valLossListY.append(round(sum(valList)/float(len(valList)),2))

      fig, ax1 = plt.subplots( nrows=1, ncols=1 ) 
      ax1.plot(trainLossListX, trainLossListY, label="Training Loss", color="red")
      ax1.plot(valLossListX, valLossListY, label="Validation Loss", color="green")
      ax1.set_ylim(bottom=0)
      ax1.legend(loc='upper right')
      ax2 = ax1.twinx()
      ax2.plot(accListX, accListY, label="Accuracy mAP (%)", color="blue")
      ax2.set_ylim(bottom=0)
      ax2.legend(loc='upper left')
      fig.savefig('test.png')  
      plt.close(fig) 

    if accListY[-1] > 15:
      params = list(net.parameters())
      optimizer = torch.optim.SGD(params, lr=0.00001, momentum=0.9, weight_decay=0.005)

    print("Iteration: " + str(x))
    print("Current train loss:\t" + str(trainLossListY[-1]))
    print("Current val loss:\t" + str(valLossListY[-1]))
    print("Current accuracy:\t" + str(mAP) + "%")
    print("")

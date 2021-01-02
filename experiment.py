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

import active_vision_dataset_processing.data_loading.active_vision_dataset as AVD 

fileName = str(random.randint(0, 1000000))
print("Filename: " + fileName)

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

def writeForPASCALVOC(path, filename, catName, data):
  outStr = catName + " "
  for i in range(len(data)):
    outStr += str(float(data[i])) + " "
  with open(path + filename + ".txt", 'a') as out:
      out.write(outStr + '\n') 

def load_synth_image():

  pathToFolder = '/content/drive/My Drive/Data/SyntheticDataset/'

  f=open(pathToFolder + "info.csv")
  lines=f.readlines()
  print("num of data: " + str(len(lines)))

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

  return image, bbox, target1, target2, image_path

def eval_images(net):

  print("Start eval")

  pathToBackgrounds = '/content/drive/My Drive/ActiveVisionDataset/'
  pathToGT = '/content/drive/My Drive/Data/GT/'
  pathToTargets = '/content/drive/My Drive/Data/AVD_and_BigBIRD_targets_v1/'

  valid_files = find_files(pathToBackgrounds, ".jpg")
  print("Files read")

  score = 0
  numOfImages = 0
  numToEval = 100
  numCorrect = 0
  fail = 0
  corr = 0

  countDict = {}
  difficulties = [3]
  #perCat = 50

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
        print(category_id)

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

        im_data = cv2.resize(pre_load_image, (1920, 1080), interpolation = cv2.INTER_AREA)
        gt_boxes = [bb_data[0], bb_data[1], bb_data[2], bb_data[3], 1]
        difficulty = 0
        if gt_boxes[2] > 300 and gt_boxes[3] > 100:
            difficulty = 1;
        elif gt_boxes[2] > 200 and gt_boxes[3] > 75:
            difficulty = 2;
        elif gt_boxes[2] > 100 and gt_boxes[3] > 50:
            difficulty = 3;
        elif gt_boxes[2] > 50 and gt_boxes[3] > 30:
            difficulty = 4;
        else:
            difficulty = 5;
        if difficulty not in difficulties:
          continue
        if difficulty not in countDict:
          countDict[difficulty] = 0
        if countDict[difficulty] >= perCat or difficulty not in countDict:
          continue
        done = True
        for key, value in countDict.iteritems():
          if value < (perCat - 1):
            done = False
        if done:
          print("Completed")
          break
        countDict[difficulty] += 1
        target1 = cv2.resize(pre_load_target_1, (80, 80), interpolation = cv2.INTER_AREA)
        target2 = cv2.resize(pre_load_target_2, (80, 80), interpolation = cv2.INTER_AREA)

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

        if len(fg_dets) < 5:
          numOfImages += 1
          fail += 1
          pass

        bb_data[2] += bb_data[0]
        bb_data[3] += bb_data[1]

        #print("Data")
        #print(fg_boxes[0])
        #print(bb_data)

        for x in range(5):

          detection = [fg_dets[x][4], fg_dets[x][0], fg_dets[x][1], fg_dets[x][2], fg_dets[x][3]]

          writeForPASCALVOC("Object-Detection-Metrics/detections/", fileName, str(category_id), detection)
          writeForPASCALVOC("Object-Detection-Metrics/groundtruths/", fileName, str(category_id), bb_data)

        numOfImages += 1
        print("Number evaluated: " + str(numOfImages))
        
        if numOfImages >= numToEval:
          break
      
      except Exception as e:
        print(e)
        pass
      
  print("numCorrect:", numCorrect)
  print("numOfImages:", numOfImages)
  print("corr:", corr)
  print("total:", corr+fail)
  print("Final score: " + str(numCorrect/numOfImages))

def eval_synth_images(net):

  score = 0
  numOfImages = 0
  numToEval = 100
  iouTot = 0

  while True:

    try:

      batch_im_data = []
      batch_target_data = []
      batch_gt_boxes = []

      im_data, gt_boxes, target1, target2, image_path = load_synth_image()

      gt_boxes[2] += gt_boxes[0]
      gt_boxes[3] += gt_boxes[1]
      
      gt = []
      gt.append(gt_boxes[0])
      gt.append(gt_boxes[1])
      gt.append(gt_boxes[2])
      gt.append(gt_boxes[3])

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

      print("box 0")
      print(boxes[0])
      print(scores[0])
      print("box 2")
      print(boxes[2])
      print(scores[2])
      print("box 50")
      print(boxes[50])
      print(scores[50])

      print(len(boxes))
      inds = np.where(scores[:, 1] > 0.1)[0]
      fg_scores = scores[inds, 1]
      fg_boxes = boxes[inds,:]
      print(len(fg_boxes))
      fg_dets = np.hstack((fg_boxes, fg_scores[:, np.newaxis])) \
          .astype(np.float32, copy=False)
      keep = nms(fg_dets, 0.7)
      fg_dets = fg_dets[keep, :]
      print(len(fg_dets))
      print(fg_dets[0][4])

      if len(fg_dets) < 5:
        continue

      im = np.array(Image.open(image_path), dtype=np.uint8)
      fig,ax = plt.subplots(1)
      ax.imshow(im)
      for i in range(len(fg_dets)-1, 0, -1):
        x1, y1, x2, y2 = fg_dets[i][0], fg_dets[i][1], fg_dets[i][2], fg_dets[i][3]
        col = fg_dets[i][4]**2
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=None, alpha=1, edgecolor=(1, 1-col, 1-col)))
      print("draw")
      x1, y1, x2, y2 = gt[0], gt[1], gt[2], gt[3]
      ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=None, alpha=1, edgecolor='b'))
      plt.savefig("1.png")

      im = np.array(Image.open(image_path), dtype=np.uint8)
      fig,ax = plt.subplots(1)
      ax.imshow(im)
      for i in range(5):
        x1, y1, x2, y2 = fg_dets[i][0], fg_dets[i][1], fg_dets[i][2], fg_dets[i][3]
        col = fg_dets[i][4]**2
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=None, alpha=1, edgecolor=(1, 1-col, 1-col)))
      print("draw")
      x1, y1, x2, y2 = gt[0], gt[1], gt[2], gt[3]
      ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=None, alpha=1, edgecolor='b'))
      plt.savefig("2.png")

      input("wait")

      #Hardcode cat ID
      category_id = 1

      for x in range(5):

        detection = [fg_dets[x][4], fg_dets[x][0], fg_dets[x][1], fg_dets[x][2], fg_dets[x][3]]

        writeForPASCALVOC("Object-Detection-Metrics/detections/", fileName, str(category_id), detection)
        writeForPASCALVOC("Object-Detection-Metrics/groundtruths/", fileName, str(category_id), gt)

      numOfImages += 1
      print("Number evaluated: " + str(numOfImages))
      
      if numOfImages >= numToEval:
        break
    
    except Exception as e:
      print(e)
      pass

###

loadNet = False

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

print("train")
net.train()

params = list(net.parameters())
optimizer = torch.optim.SGD(params, lr=cfg.LEARNING_RATE,
                                    momentum=cfg.MOMENTUM, 
                                    weight_decay=cfg.WEIGHT_DECAY)

trainFirst = True
verbose = False
numToTrainOn = 500

batchSize = 2

if trainFirst:

  train_loss = 0
  epoch_loss = 0
  epoch_step_cnt = 0

  for i in range(numToTrainOn):

    batch_im_data = []
    batch_target_data = []
    batch_gt_boxes = []

    for j in range(batchSize):

      if verbose:
        print("Loading image")
      im_data, gt_boxes, target1, target2, image_path = load_synth_image()

      gt_boxes[2] += gt_boxes[0]
      gt_boxes[3] += gt_boxes[1]

      gt_boxes = np.asarray([[gt_boxes[0],gt_boxes[1],gt_boxes[2],gt_boxes[3],1]])

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

    net(target_data, im_data, im_info, gt_boxes=gt_boxes)
    loss = net.loss

    optimizer.zero_grad()
    loss.backward()
    clip_gradient(net, 10.)
    optimizer.step()

    train_loss += loss.data[0]
    epoch_step_cnt += 1
    epoch_loss += loss.data[0]

    print("loss: " + str(loss.data[0]))
    print("Numbers trained: " + str(i))
    print("")

    if (loss.data[0] < 0.1):
      break

print("Eval images")
net.eval()
eval_synth_images(net)

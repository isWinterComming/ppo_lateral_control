#!/usr/bin/env python3
import sys
import os
sys.path.append(os.getcwd())
import glob
import numpy as np
import random
import cv2
from tqdm import tqdm

from common.transformations.camera import get_view_frame_from_road_frame

import torch.nn as nn
import torch
import torch.nn.functional as F
import random
cv2.ocl.setUseOpenCL(True)






def reshape_yuv(frames):
  H = (frames.shape[0]*2)//3
  W = frames.shape[1]
  in_img1 = np.zeros((6, H//2, W//2), dtype=np.uint8)

  in_img1[0] = frames[0:H:2, 0::2]
  in_img1[1] = frames[1:H:2, 0::2]
  in_img1[2] = frames[0:H:2, 1::2]
  in_img1[3] = frames[1:H:2, 1::2]
  in_img1[4] = frames[H:H+H//4].reshape((H//2, W//2))
  in_img1[5] = frames[H+H//4:H+H//2].reshape((H//2, W//2))
  return in_img1

def to_rgb(img):
    H = 256
    W = 512

    frames = np.zeros((H*3//2, W), dtype=np.uint8)
    frames[0:H:2, 0::2] = img[0]
    frames[1:H:2, 0::2] = img[1]
    frames[0:H:2, 1::2] = img[2]
    frames[1:H:2, 1::2] = img[3]
    frames[H:H+H//4] = img[4].reshape((-1, H//4, W))
    frames[H+H//4:H+H//2] = img[5].reshape((-1, H//4, W))

    return cv2.cvtColor(frames, cv2.COLOR_YUV2BGR_I420)


def get_calib_matrix(is_c2=1):

  if is_c2==1:
    rot_angle =  [-0.0, 0.05, -0.0, 1.22, -0.]
    cam_insmatrixs = np.array([[910.0,  0.0,   0.5 * 1164],
                            [0.0,  910.0,   0.5 * 874],
                            [0.0,  0.0,     1.0]])
  elif is_c2==2:

    rot_angle =  [-0.0005339412952404624, 0.0434282135994484, -0.02439150642603636, 1.35, -0.06]
    cam_insmatrixs = np.array([[2648.0,   0.,   1928/2.],
                            [0.,  2648.0,  1208/2.],
                            [0.,    0.,     1.]])
  else:
    # rot_angle =  [-0.0, 0.05, -0.0, 1.22, -0.]
    # cam_insmatrixs = np.array([[910.0,  0.0,   0.5 * 1280],
    #                 [0.0,  910.0,   0.5 * 1000],
    #                 [0.0,  0.0,     1.0]])
    rot_angle =  [-0.0, -0.04, 0., 1.22, -0.]
    cam_insmatrixs = np.array([[2000.0,  0.0,   0.5 * 1920],
                    [0.0,  2000.0,   0.5 * 1080],
                    [0.0,  0.0,     1.0]])
  ang_x = rot_angle[0]
  ang_y = rot_angle[1] 
  ang_z = rot_angle[2] + np.clip(np.random.normal(loc=0.0, scale=0.005), -0.02, 0.02)

  dev_height = rot_angle[3]
  lat_bias = rot_angle[4] + np.clip(np.random.normal(loc=0.0, scale=0.1), -0.4, 0.4)
    

  MEDMDL_INSMATRIX = np.array([[910.0,  0.0,   0.5 * 512],
                                  [0.0,  910.0,   47.6],
                                  [0.0,  0.0,     1.0]])
  camera_frame_from_ground = np.dot(cam_insmatrixs,
                                      get_view_frame_from_road_frame(ang_x, ang_y, ang_z, dev_height, lat_bias))[:, (0, 1, 3)]
  calib_frame_from_ground = np.dot(MEDMDL_INSMATRIX,
                                      get_view_frame_from_road_frame(0, 0, 0, 1.22))[:, (0, 1, 3)]
  calib_msg = np.dot(camera_frame_from_ground, np.linalg.inv(calib_frame_from_ground))


  return calib_msg


import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor


def read_image_worker(path):
    return cv2.imread(path)


# 多线程读取（适合IO密集型）
def batch_read_multithread(image_paths, max_workers=24):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(read_image_worker, image_paths))
    return [img for img in results if img is not None]

# 多进程读取（适合CPU密集型）
def batch_read_multiprocess(image_paths, max_workers=24):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(read_image_worker, image_paths))
    return [img for img in results]
def datagen(bat_size = 32, is_rgb = False):
  files_length = len(img_datasets)
  sample_list = [i for i in range(files_length)]

  while True:
    try:
      st =time.time()
      if len(sample_list) < bat_size:
        print(len(sample_list))
        sample_list = [i for i in range(files_length)]
      if is_rgb:
          bat_imgs = np.zeros((bat_size, 3, 128, 256), dtype=np.float32)
      else:
          bat_imgs = np.zeros((bat_size, 6, 128, 256), dtype=np.float32)

      file_indexs = random.sample(sample_list,bat_size)
      file_paths = [img_datasets[i] for i in file_indexs]
      sample_imgs = batch_read_multithread(file_paths)
      for j in range(bat_size):
        sample_list.remove(file_indexs[j])
        img = sample_imgs[j]
        is_c2 = "imgs_c2" in file_paths[j]

        if "imgs_c2" in file_paths[j]:
            is_c2 = 1
        elif "imgs_c3" in file_paths[j]:
            is_c2 = 2
        else:
            is_c2 = 0
          
        # print(img_name.split("/")[1][0])
        new_calib_matrix = get_calib_matrix(is_c2)
        img_bgr = cv2.warpPerspective(src=img, M=new_calib_matrix, dsize=(512,256), flags=cv2.WARP_INVERSE_MAP)

        img_bgr = img_bgr[:,::-1,:] if np.random.rand() > 0.5 else  img_bgr # random rever image with u axis
        if is_rgb:
          img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
          bat_imgs[j] = cv2.resize(img, (256, 128), interpolation=cv2.INTER_CUBIC).transpose(2,0,1)
        else:
          img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)
          bat_imgs[j] = reshape_yuv(img)

      print(time.time() - st, bat_imgs.shape)

      yield (bat_imgs, len(sample_list) < bat_size)
    except KeyboardInterrupt:
      raise
    except:
      traceback.print_exc()
      pass
img_datasets = glob.glob(f"big_imgs/*/*.png")
total_data_size = int(len(img_datasets))

# dd = datagen()

# while True:
#   tx, vaalid = next(dd)
#   print('s')
#   # print()
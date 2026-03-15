#!/usr/bin/env python3
import argparse
import math
import threading
import time
import os
from multiprocessing import Process, Queue
from typing import Any
import logging
import cv2
import scipy as sp

import numpy as np
from numpy import random
import socket
import struct
import json
from lib.realtime import Ratekeeper
from lib.pid import PIController
from train_mae_cnn import *

import time
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from metadrive import MetaDriveEnv

from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE
import matplotlib.pyplot as plt
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.utils.math import clip, norm


from metadrive.component.sensors.base_camera import _cuda_enable
from metadrive.component.map.pg_map import MapGenerateMethod
from panda3d.core import Vec3, Texture, GraphicsOutput


from control.lat_mpc import LatMpc
import math
controller = LatMpc()

W = 1164
H = 874

C3_POSITION = Vec3(0, 0, 1.22)


class CopyRamRGBCamera(RGBCamera):
  """Camera which copies its content into RAM during the render process, for faster image grabbing."""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cpu_texture = Texture()
    self.buffer.addRenderTexture(self.cpu_texture, GraphicsOutput.RTMCopyRam)

  def get_rgb_array_cpu(self):
    origin_img = self.cpu_texture
    img = np.frombuffer(origin_img.getRamImage().getData(), dtype=np.uint8)
    img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), -1))
    img = img[:,:,:3] # RGBA to RGB
    # img = np.swapaxes(img, 1, 0)
    img = img[::-1] # Flip on vertical axis
    return img


class RGBCameraWide(CopyRamRGBCamera):
  def __init__(self, *args, **kwargs):
    super(RGBCameraWide, self).__init__(*args, **kwargs)
    cam = self.get_cam()
    cam.setPos(C3_POSITION)
    
    lens = self.get_lens()
    lens.setFov(120)
    lens.setNear(0.1)

class RGBCameraRoad(CopyRamRGBCamera):
  def __init__(self, *args, **kwargs):
    super(RGBCameraRoad, self).__init__(*args, **kwargs)
    cam = self.get_cam()
    cam.setPos(C3_POSITION)
    lens = self.get_lens()
    lens.setFocalLength(910)
    # print(dir(lens))
    lens.setFov(40)
    lens.setNear(0.1)


def straight_block(length):
  return {
    "id": "S",
    "pre_block_socket_index": 0,
    "length": length
  }

def curve_block(length, angle=45, direction=0):
  return {
    "id": "C",
    "pre_block_socket_index": 0,
    "length": length,
    "radius": length,
    "angle": angle,
    "dir": direction
  }


from help_cv import reshape_yuv, draw_path, PlanModelV2, get_calib_matrix, ANCHOR_TIME



def main():
  plan_model = PlanModelV2()
  sensors = {
    "rgb_road": (RGBCameraRoad, W, H, )
  }
  config = dict(
    use_render=False,
    vehicle_config=dict(
      enable_reverse=False,
      image_source="rgb_road",
      spawn_longitude=15
    ),
    sensors=sensors,
    image_on_cuda=False,
    image_observation=True,
    interface_panel=[],
    out_of_route_done=False,
    on_continuous_line_done=False,
    crash_vehicle_done=False,
    crash_object_done=False,
    traffic_density=0.0, # traffic is incredibly expensive
    map_config=dict(
      type=MapGenerateMethod.PG_MAP_FILE,
      config=[
        None,
        straight_block(120),
        curve_block(240, 90),
        straight_block(120),
        curve_block(240, 90),
        straight_block(120),
        curve_block(240, 90),
        straight_block(120),
        curve_block(240, 90),
      ]
    ),
    decision_repeat=1,
    physics_world_step_size=5./100,
    preload_models=False
  )
  env = MetaDriveEnv(config)
  env.reset()
  env.vehicle.expert_takeover = False


  def get_cam_as_rgb(cam):
    cam = env.engine.sensors[cam]
    img = cam.perceive(env.vehicle)
    #print(img)
    return img
  

  lateralActive = False
  LongitudinalActive = False
  is_openpilot_engaged = False

  throttle_out = steer_out = brake_out = 0
  throttle_manual = steer_manual = brake_manual = 0


  throttle_manual_multiplier = 1  # keyboard signal is always 1
  brake_manual_multiplier = 1  # keyboard signal is always 1
  steer_manual_multiplier = 10   # keyboard signal is always 1


  # accel controller
  ACCEL_CTRL_KP_X = [0, 10, 20, 30, 40]
  ACCEL_CTRL_KP_Y = [0.2, 0.25, 0.25, 0.25, 0.25]

  ACCEL_CTRL_KI_X = [0, 10, 20, 30, 40]
  ACCEL_CTRL_KI_Y = [0.02, 0.02, 0.02, 0.02, 0.02]

  accelController = PIController([ACCEL_CTRL_KP_X, ACCEL_CTRL_KP_Y], [ACCEL_CTRL_KI_X, ACCEL_CTRL_KI_Y], \
                          pos_limit=1, neg_limit=-1, rate=20, sat_limit=0.8)
  accelController.reset()

 
  rk = Ratekeeper(20, print_delay_threshold=0.04)

  road_image = np.zeros((H,W,3), dtype=np.float32)
  last_bat_img = None
  # main loop
  while True:
    

    road_image[...] = get_cam_as_rgb('rgb_road')
    print(road_image.shape)


    calib_msg = get_calib_matrix( pos_bias=0, theta_bias=0, 
                     ang_x=0, ang_y=0.13, ang_z=0, dev_height=1.22, lat_bias=0)

    img_bgr = cv2.warpPerspective(src=road_image, M=calib_msg, dsize=(512,256), flags=cv2.WARP_INVERSE_MAP)
    # img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)
    # feed_imgs = reshape_yuv(img)[None,:,:,:]
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 128), interpolation=cv2.INTER_CUBIC)
    feed_imgs = img.transpose(2,0,1).reshape(1, 3, 128, 256)
    
    # use last image and current image to feed model, scalar to [0 1]
    traj_batch = plan_model.run(feed_imgs.astype(np.float32) * 255 )


    ## draw pred multi-trajectory
    best_idx = 0
    best_prob = 0.0
    idx = 0
    for traj in traj_batch:
      if traj[0] > best_prob:
        best_idx = idx
        best_prob = traj[0]
        
      idx +=1

    ## controller, from plan path to angle
    # best_idx = 0
    traj_x = traj_batch[best_idx][1]
    traj_y = -1*traj_batch[best_idx][2]

    pred_dx = np.clip(traj_x[1:33] - traj_x[0:32], 1e-1, 1e4) # avoid zero divide
    pred_dy = traj_y[1:33] - traj_y[0:32]
    traj_theta = np.array([math.atan2(y, x) for x,y in zip(pred_dx.flatten(), pred_dy.flatten())]) # rad

    velocity = np.sqrt(env.vehicle.velocity[0]**2 + env.vehicle.velocity[1]**2)
    
    controller.update(True, velocity, 0, traj_x.flatten(), traj_y.flatten(), traj_theta.flatten(), ANCHOR_TIME.flatten())

    device_path = np.concatenate([traj_batch[best_idx][1].reshape(1,33), traj_batch[best_idx][2].reshape(1,33), np.zeros((1,33))], axis=0)
    draw_path(device_path.T, img_bgr, width=0.2, height=1.22)

    cv2.imshow("grab_img", img_bgr)
    cv2.waitKey(2)


    # apply control commands
    vc = [controller.steer_out/(env.vehicle.MAX_STEERING), 0.5]
    print(vc, env.vehicle.velocity)
    o, r, tm, tc, info = env.step(vc)

    rk.keep_time()


main()

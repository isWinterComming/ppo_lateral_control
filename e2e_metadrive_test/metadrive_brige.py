#!/usr/bin/env python3
import argparse
import math
from multiprocessing import Process, Queue
from typing import Any
import cv2
import numpy as np
import time
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.component.map.pg_map import MapGenerateMethod
from panda3d.core import Vec3, Texture, GraphicsOutput
import matplotlib.pyplot as plt

from lib.realtime import Ratekeeper
from lib.pid import PIController
from help_cv import reshape_yuv, draw_path, PlanModelV3, get_calib_matrix, ANCHOR_TIME, PlanModel, LowpassFilter


## Camera parameters.
FRAME_WIDTH = 1164
FRAME_HEIGHT = 874
CAMERA_HEIGHT = 1.22 # m.
C3_POSITION = Vec3(0.0, 0, CAMERA_HEIGHT)
C3_HPR = Vec3(0, 0,0)


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

class RGBCameraRoad(CopyRamRGBCamera):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    lens = self.get_lens()
    lens.setFov(65.3)
    lens.setNear(0.1)


def straight_block(length):
  return {
    "id": "S",
    "pre_block_socket_index": 0,
    "length": length
  }

def curve_block(length, angle=45, direction=1):
  return {
    "id": "C",
    "pre_block_socket_index": 0,
    "length": length,
    "radius": length,
    "angle": angle,
    "dir": 0
  }
  

def main():
  sensors = {
    "rgb_road": (RGBCameraRoad, FRAME_WIDTH, FRAME_HEIGHT, )
  }

  radius = 500
  config = dict(
    use_render=False,
    vehicle_config=dict(
      enable_reverse=False,
         spawn_position_heading=[[-20., 4], 0],
      image_source="rgb_road",
      spawn_longitude=15,
      spawn_velocity=[15,0],
      
    ),
    sensors=sensors,
    image_on_cuda=False,
    image_observation=True,
    interface_panel=[],
    out_of_route_done=False,
    on_continuous_line_done=False,
    crash_vehicle_done=False,
    crash_object_done=False,
    # arrive_dest_done=False,
    traffic_density=0.0, # traffic is incredibly expensive
    map_config=dict(
      type=MapGenerateMethod.PG_MAP_FILE,
      lane_num=3,
      lane_width=3.8,
        config=[
          None,
          straight_block(50),
          curve_block(radius, 180),
          # straight_block(2),
          # curve_block(radius, 90),
          # straight_block(2),
          # curve_block(radius, 90),
          straight_block(50),
          curve_block(radius, 180),
        ]
    ), 
    # num_scenarios=100,    # 生成1000个不同的场景，存一个库里
    # start_seed=2,       # 不指定种子，每次启动都从系统取随机数
    # random_lane_width=True, # 随机车道宽度
    # random_lane_num=True,    # 随机车道数量

    decision_repeat=1,
    physics_world_step_size=5/100,
    preload_models=False,
    show_logo=False,
    # anisotropic_filtering=False
  )

  env = MetaDriveEnv(config)
  env.reset()
  # env.vehicle.expert_takeover = False

  def get_cam_as_rgb(cam):
    cam = env.engine.sensors[cam]
    cam.get_cam().reparentTo(env.vehicle.origin)
    cam.get_cam().setPos(C3_POSITION)
    cam.get_cam().setHpr(C3_HPR)
    img = cam.perceive(to_float=False)
    if not isinstance(img, np.ndarray):
      img = img.get() # convert cupy array to numpy
    return img

  # Accel controller.
  ACCEL_CTRL_KP_X = [0, 10, 20, 30, 40]
  ACCEL_CTRL_KP_Y = [0.2, 0.25, 0.25, 0.25, 0.25]
  ACCEL_CTRL_KI_X = [0, 10, 20, 30, 40]
  ACCEL_CTRL_KI_Y = [0.02, 0.02, 0.02, 0.02, 0.02]
  # Speed controller.
  speed_controller = PIController([ACCEL_CTRL_KP_X, ACCEL_CTRL_KP_Y], [ACCEL_CTRL_KI_X, ACCEL_CTRL_KI_Y], \
                          pos_limit=1, neg_limit=-1, rate=20, sat_limit=0.8)
  speed_controller.reset()

  # Rate kepper.
  rk = Ratekeeper(20, print_delay_threshold=0.04)
  
  plan_model = PlanModel()
  out = cv2.VideoWriter('output.mp4',  cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (512, 256))

  last_heading = None
  yaw_rate = LowpassFilter(sample_freq=20, cutoff_freq=5)

  speed_cmds = []
  speed_states = []

  yr_cmds = []
  yr_states = []

  stop_time = 120.

  # main loop
  while True:

    heading_dir_now = np.array(env.vehicle.heading).reshape(1,2)
    # print(np.array(heading_dir_now), heading_dir_last)
    if last_heading is not None:
      heading_dir_last = np.array(last_heading).reshape(1,2)
      cos_beta = (heading_dir_now @ heading_dir_last.T) / ( np.linalg.norm(heading_dir_now) * np.linalg.norm(heading_dir_last))
      sin_beta = (np.cross(heading_dir_now, heading_dir_last) ) / np.linalg.norm(heading_dir_now) * np.linalg.norm(heading_dir_last)
      # beta_diff = np.arccos(clip(cos_beta, -1.0, 1.0))
      beta_diff = -1*math.atan2(sin_beta, cos_beta)
      yaw_rate_raw = yaw_rate.update(beta_diff / 0.05)
    else:
      yaw_rate_raw = 0

    last_heading = heading_dir_now
    yaw_rate_raw = yaw_rate_raw * 180/np.pi # degps
    print('yaw_rate_raw', yaw_rate_raw)


    st = time.time()
    road_image = get_cam_as_rgb('rgb_road')
    calib_msg = get_calib_matrix( pos_bias=0.00, theta_bias=0, 
                     ang_x=0, ang_y=0.00, ang_z=0, dev_height=1.6, lat_bias=0)
    # print(calib_msg)
    img_bgr = cv2.warpPerspective(
        src=road_image,
        M=calib_msg,
        dsize=(512, 256),
        flags=cv2.WARP_INVERSE_MAP,
    )
    del road_image
    
    # to model frame like and run model.
    traj_pred, vege = plan_model.run(img_bgr )

    ## controller, from plan path to angle
    traj_x = traj_pred[0,:,0]
    traj_y = -1*traj_pred[0,:,1]
    pred_dx = np.clip(traj_x[1:33] - traj_x[0:32], 1e-1, 1e4) # avoid zero divide
    pred_dy = traj_y[1:33] - traj_y[0:32]
    traj_theta = np.array([math.atan2(y, x) for x,y in zip(pred_dx.flatten(), pred_dy.flatten())]) # rad
    velocity = np.sqrt(env.vehicle.velocity[0]**2 + env.vehicle.velocity[1]**2)
    desired_yawrate = plan_model.mpc_controller.update(True, velocity, yaw_rate_raw, traj_x.flatten(), traj_y.flatten(), traj_theta.flatten(), ANCHOR_TIME.flatten())

    ## model trajectorh visual to image pixs space.
    device_path = np.concatenate([traj_pred[0,:,0].reshape(1,33), traj_pred[0,:,1].reshape(1,33), np.zeros((1,33))], axis=0)
    draw_path(device_path.T, img_bgr, width=0.5, height=1.2, fill_color=(255,144,30))
    
    ## get constant speed controller command.
    speed_cmd = 22.0
    gas_brake_out = speed_controller.update(speed_cmd, velocity, 1, speed=velocity, 
                          check_saturation=True, override=False, feedforward=0.0, deadzone=0., freeze_integrator=False)

    steer_out = plan_model.mpc_controller.steer_out if rk._frame > 50 else 0.0
    desired_yawrate = desired_yawrate if rk._frame > 50 else 0.0
    # apply control commands
    vc = [steer_out/(env.vehicle.MAX_STEERING), gas_brake_out]
    print(vc, velocity, time.time() - st)
    st = time.time()
    o, r, tm, tc, info = env.step(vc)
    # print('env step conmsume time is : ', time.time() - st)

    cv2.imshow('test', img_bgr)
    cv2.waitKey(1)
    rk.keep_time()
    out.write(img_bgr)
    cv2.imwrite(f'./images/test_img_{rk.frame}.png', img_bgr)

    speed_cmds.append(vege)
    speed_states.append(velocity)
    yr_cmds.append(desired_yawrate*180./np.pi)
    yr_states.append(yaw_rate_raw)

    if rk._frame > stop_time * 20:
      break
  plt.subplot(2,1,1)
  plt.plot(yr_cmds, 'g--', label='yr_cmds')
  plt.plot(yr_states, 'r--', label='yr_states')
  plt.legend()
  plt.subplot(2,1,2)
  plt.plot(speed_cmds, 'g--', label='speed_cmds')
  plt.plot(speed_states, 'r--', label='speed_states')
  plt.legend()
  plt.show()

main()
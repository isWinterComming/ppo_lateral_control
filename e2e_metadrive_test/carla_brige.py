#!/usr/bin/env python3
import argparse
from distutils.spawn import spawn
import math
import threading
import time
import os
from multiprocessing import Process, Queue
from typing import Any
import logging
import cv2
import scipy as sp

import carla  # pylint: disable=import-error
from carla import VehicleLightState as vls
from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error


import numpy as np
from numpy import random
import socket
import struct
import json
from tools.sim.carla_simulink import CarlaSimulink
from lib.realtime import Ratekeeper
from lib.pid import PIController
from carla_lib import *


# parameters
parser = argparse.ArgumentParser(description='Bridge between CARLA and openpilot.')
parser.add_argument('--joystick', action='store_true')
parser.add_argument('--low_quality', default=True, action='store_true')
parser.add_argument('--town', type=str, default='Town04_Opt')
parser.add_argument('--spawn_point', dest='num_selected_spawn_point', type=int, default=6)

#------------------------
#--generate traffic args--
#------------------------
parser.add_argument(
    '-n', '--number-of-vehicles',
    metavar='N',
    default=200,
    type=int,
    help='Number of vehicles (default: 30)')
parser.add_argument(
    '--safe',
    action='store_true',
    help='Avoid spawning vehicles prone to accidents')
parser.add_argument(
    '--autopilot',
    action='store_true',
    help='Avoid spawning vehicles prone to accidents')
parser.add_argument(
    '--filterv',
    metavar='PATTERN',
    default='vehicle.*',
    help='Filter vehicle model (default: "vehicle.*")')
parser.add_argument(
    '--generationv',
    metavar='G',
    default='All',
    help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
parser.add_argument(
    '--tm-port',
    metavar='P',
    default=8000,
    type=int,
    help='Port to communicate with TM (default: 8000)')
parser.add_argument(
    '--asynch',
    action='store_true',
    help='Activate asynchronous mode execution')
parser.add_argument(
    '--hybrid',
    action='store_true',
    help='Activate hybrid mode for Traffic Manager')
parser.add_argument(
    '-s', '--seed',
    metavar='S',
    type=int,
    help='Set random device seed and deterministic mode for Traffic Manager')
parser.add_argument(
    '--car-lights-on',
    action='store_true',
    default=True,
    help='Enable car lights')
parser.add_argument(
    '--hero',
    action='store_true',
    default=False,
    help='Set one of the vehicles as hero')
parser.add_argument(
    '--respawn',
    action='store_true',
    default=False,
    help='Automatically respawn dormant vehicles (only in large maps)')
parser.add_argument(
    '--no-rendering',
    action='store_true',
    default=False,
    help='Activate no rendering mode')
#-------------------------------------------------

args = parser.parse_args()

W, H = 1928, 1208
REPEAT_COUNTER = 5
PRINT_DECIMATION = 10
STEER_RATIO = 16.2

def update_world(word):
    st = time.time()
    word.tick()
    dt = time.time() - st
    if dt < 0.05:
      time.sleep(0.05 - dt)

def bridge(q):
  # setup CARLA
  client = carla.Client("127.0.0.1", 2000)
  client.set_timeout(100.0)

  # setup pnc+carla lib
  #pncCarla = CarlaPnc()
  

  world = client.load_world(args.town)
  print('connected from server')

  print(client.get_available_maps())

  settings = world.get_settings()
  settings.synchronous_mode = True # Enables synchronous mode
  settings.fixed_delta_seconds = 0.05
  world.apply_settings(settings)


  #world.set_weather(carla.WeatherParameters.HarainNoon)

  if args.low_quality:
    world.unload_map_layer(carla.MapLayer.Foliage)
    world.unload_map_layer(carla.MapLayer.Buildings)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    world.unload_map_layer(carla.MapLayer.Props)
    world.unload_map_layer(carla.MapLayer.StreetLights)
    world.unload_map_layer(carla.MapLayer.Particles)

  blueprint_library = world.get_blueprint_library()

  world_map = world.get_map()

  spawn_points = world_map.get_spawn_points()

  print(len(spawn_points))

  #-------------generate traffic------------------
  #modified by lilinfneg-2022-2-14
  traffic_manager = client.get_trafficmanager(args.tm_port)
  traffic_manager.set_global_distance_to_leading_vehicle(2.5)
 

  if args.respawn:
      traffic_manager.set_respawn_dormant_vehicles(True)
  if args.hybrid:
      pass
      #traffic_manager.set_hybrid_physics_mode(True)
      #traffic_manager.set_hybrid_physics_radius(70.0)
  if args.seed is not None:
      traffic_manager.set_random_device_seed(args.seed)
  settings = world.get_settings()
  if not args.asynch:
      traffic_manager.set_synchronous_mode(True)
      if not settings.synchronous_mode:
          synchronous_master = True
          settings.synchronous_mode = True
          settings.fixed_delta_seconds = 0.05
      else:
          synchronous_master = False
  else:
      print("You are currently in asynchronous mode. If this is a traffic simulation, \
      you could experience some issues. If it's not working correctly, switch to synchronous \
      mode by using traffic_manager.set_synchronous_mode(True)")
  if args.no_rendering:
      settings.no_rendering_mode = True
  world.apply_settings(settings)
  blueprints = get_actor_blueprints(world, args.filterv, args.generationv)

  #print(blueprints)
  if args.safe:
      blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
      blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
      blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
      blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
      blueprints = [x for x in blueprints if not x.id.endswith('t2')]
      blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
      blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
      blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]
  blueprints = sorted(blueprints, key=lambda bp: bp.id)
  spawn_points = world.get_map().get_spawn_points()

  print(len(spawn_points))
  number_of_spawn_points = len(spawn_points)
  if args.number_of_vehicles < number_of_spawn_points:
      random.shuffle(spawn_points)
  elif args.number_of_vehicles > number_of_spawn_points:
      msg = 'requested %d vehicles, but could only find %d spawn points'
      logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
      args.number_of_vehicles = number_of_spawn_points
  # @todo cannot import these directly.

  SpawnActor = carla.command.SpawnActor
  SetAutopilot = carla.command.SetAutopilot
  SetVehicleLightState = carla.command.SetVehicleLightState
  FutureActor = carla.command.FutureActor
  print(dir(carla.command) )

  # --------------
  # Spawn vehicles
  # --------------
  batch = []
  vehicles_list = []
  hero = args.hero
  for n, transform in enumerate(spawn_points):
    # ------------------ spawn ego car------------------- 
    if n >= args.number_of_vehicles:
      assert len(spawn_points) > args.num_selected_spawn_point, \
        f'''No spawn point {args.num_selected_spawn_point}, try a value between 0 and
        {len(spawn_points)} for this town.'''
      #print(spawn_point)
      # spawn_point = carla.Transform(carla.Location(x=-321.549286, y=8.489051, z=0.377607), \
      #       carla.Rotation(pitch=-0.431458, yaw=-179.923172, roll=0.000000))
      spawn_point = carla.Transform(carla.Location(x=9.5, y=230, z=0.2), \
          carla.Rotation(pitch=0, yaw=-90, roll=0))


      vehicle_bp = blueprint_library.filter('vehicle.tesla.*')[1]
      vehicle = world.spawn_actor(vehicle_bp, spawn_point)
      traffic_manager.update_vehicle_lights(vehicle, True)
      if args.autopilot:
        vehicle.set_autopilot()

      print(vehicle.get_transform().location.x)
      print(vehicle.get_transform().location.y)  
      break

    # ------------------ spawn surranding car-------------------
    blueprint = random.choice(blueprints)
    if blueprint.has_attribute('color'):
        color = random.choice(blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)
    if hero:
        blueprint.set_attribute('role_name', 'hero')
        hero = False
    else:
        blueprint.set_attribute('role_name', 'autopilot')
    # prepare the light state of the cars to spawn
    light_state = vls.NONE
    if args.car_lights_on:
        light_state = vls.Position | vls.LowBeam | vls.LowBeam
    # spawn the cars and set their autopilot and light state all together
    '''
    batch.append(SpawnActor(blueprint, transform)
        .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
        .then(SetVehicleLightState(FutureActor, light_state)))
    '''
    v = world.spawn_actor(blueprint, transform)
    v.set_autopilot()
    #v.set_light_state(carla.VehicleLightState.LeftBlinker)
    #v.set_light_state(carla.VehicleLightState.RightBlinker)
    #v.set_light_state()
    traffic_manager.auto_lane_change(v, True)
    #traffic_manager.update_vehicle_lights(v, True)
  

  for response in client.apply_batch_sync(batch, synchronous_master):
      if response.error:
          logging.error(response.error)
      else:
          vehicles_list.append(response.actor_id)




  max_steer_angle = vehicle.get_physics_control().wheels[0].max_steer_angle
  # make tires less slippery
  # wheel_control = carla.WheelPhysicsControl(tire_friction=5)
  physics_control = vehicle.get_physics_control()
  physics_control.mass = 2326
  # physics_control.wheels = [wheel_control]*4
  physics_control.torque_curve = [[20.0, 500.0], [5000.0, 500.0]]
  physics_control.gear_switch_time = 0.0
  vehicle.apply_physics_control(physics_control)


  vehicle_state = VehicleState()


  blueprint = blueprint_library.find('sensor.camera.rgb')
  blueprint.set_attribute('image_size_x', str(W))
  blueprint.set_attribute('image_size_y', str(H))
  blueprint.set_attribute('fov', '40')
  blueprint.set_attribute('sensor_tick', '0.05')
  transform = carla.Transform(carla.Location(x=0.8, z=1.13), carla.Rotation(yaw=0.))

  transform_top = carla.Transform(carla.Location(x=-4.8, z=1.3), carla.Rotation(yaw=180.))

  # front camera
  frontCamera = world.spawn_actor(blueprint, transform, attach_to=vehicle)
  frontCamerad = Camerad(640, 480, 'cam')
  frontCamera.listen(frontCamerad.cam_callback)


  transform1 = carla.Transform(carla.Location(x=1, z=1.3), carla.Rotation(yaw=0.))
  blueprint = blueprint_library.find('sensor.camera.rgb')
  blueprint.set_attribute('image_size_x', str(1928))
  blueprint.set_attribute('image_size_y', str(1208))
  blueprint.set_attribute('fov', str(40))
  blueprint.set_attribute('sensor_tick', str(1/20))
  blueprint.set_attribute('enable_postprocess_effects', 'True')
  camera = world.spawn_actor(blueprint, transform1, attach_to=vehicle)
  frontCamerad1 = Camerad(1928, 1208, 'cam1')
  camera.listen(frontCamerad1.cam_callback)




  # front camera
  readCamera = world.spawn_actor(blueprint, transform_top, attach_to=vehicle)
  reartCamerad = Camerad(640, 300, 'rear_cam')
  readCamera.listen(reartCamerad.cam_callback)


  # reenable IMU
  imu_bp = blueprint_library.find('sensor.other.imu')
  imu = world.spawn_actor(imu_bp, transform, attach_to=vehicle)
  imud = IMUSensor()
  imu.listen(lambda imu: imud.imu_callback(imu, vehicle_state))




  # init
  throttle_ease_out_counter = REPEAT_COUNTER
  brake_ease_out_counter = REPEAT_COUNTER
  steer_ease_out_counter = REPEAT_COUNTER

  vc = carla.VehicleControl(throttle=0, steer=0, brake=0, reverse=False)


  lateralActive = False
  LongitudinalActive = False
  is_openpilot_engaged = False
  throttle_out = steer_out = brake_out = 0
  throttle_op = steer_op = brake_op = 0
  throttle_manual = steer_manual = brake_manual = 0

  old_steer = old_brake = old_throttle = 0
  throttle_manual_multiplier = 1  # keyboard signal is always 1
  brake_manual_multiplier = 1  # keyboard signal is always 1
  steer_manual_multiplier = 10   # keyboard signal is always 1

  is_egoVehReverse = False

  keyboard_keep_cnt = 0
  cruise_button = 0

  # accel controller
  ACCEL_CTRL_KP_X = [0, 10, 20, 30, 40]
  ACCEL_CTRL_KP_Y = [0.2, 0.25, 0.25, 0.25, 0.25]

  ACCEL_CTRL_KI_X = [0, 10, 20, 30, 40]
  ACCEL_CTRL_KI_Y = [0.02, 0.02, 0.02, 0.02, 0.02]

  accelController = PIController([ACCEL_CTRL_KP_X, ACCEL_CTRL_KP_Y], [ACCEL_CTRL_KI_X, ACCEL_CTRL_KI_Y], \
                          pos_limit=1, neg_limit=-1, rate=100, sat_limit=0.8)
  accelController.reset()

  # debug udp port
  udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)     

  throttle_cmds = throttle_manual = steer_manual = brake_manual = 0.0


  # behavior
  # agent = BasicAgent(vehicle)
  # destination = random.choice(spawn_points).location
  # agent.set_destination(destination, vehicle.get_transform().location)
  for _ in range(3):
    world.tick()
  
  #agent.run_step()
  #print(len(agent._local_planner._waypoints_queue))


  # global_refer_points = agent._local_planner._waypoints_queue
  # print(global_refer_points[0][0].transform.location.x)
  # print(global_refer_points[0][0].transform.location.y)
  # print(global_refer_points[-1][0].transform.location.x)
  # print(global_refer_points[-1][0].transform.location.y)

  print(vehicle.get_transform().location.x)
  print(vehicle.get_transform().location.y)  
  #time.sleep(10)
  carlaSimulink = CarlaSimulink()
  
  # can loop
  print('start can')
  rk = Ratekeeper(100, print_delay_threshold=0.04)

  # main loop
  num = 5
  while True:
    t = time.time()
    
    # 1. Read the throttle, steer and brake from op or manual controls
    # 2. Set instructions in Carla
    # 3. Send current carstate to op via can

    
    throttle_out = steer_out = brake_out = 0.0
    throttle_op = steer_op = brake_op = 0
    

    # --------------Step 1-------------------------------
    if not q.empty():
      message = q.get()
      m = message.split('_')
      num = 5
      #print(m)
      if m[0] == "steer":
        steer_manual += float(m[1])/1.0
        steer_manual = clip( steer_manual, -1*max_steer_angle, max_steer_angle)
        is_openpilot_engaged = False

      elif m[0] == "throttle":
        throttle_cmds += float(m[1])
        throttle_cmds = clip( throttle_cmds, -1.0, 1.0)
        is_openpilot_engaged = False
        throttle_manual = clip( throttle_cmds, 0.0, 1.0)
        brake_manual = clip(-throttle_cmds, 0.0, 1.0)

      elif m[0] == "reverse":
        throttle_cmds = throttle_manual = steer_manual = brake_manual = 0.0
        is_egoVehReverse = not is_egoVehReverse
        is_openpilot_engaged = False

      elif m[0] == "cruise":
        # reset manual control
        throttle_cmds = throttle_manual = steer_manual = brake_manual = 0.0
        if m[1] == "down":
          cruise_button = CruiseButtons.DECEL_SET
          is_openpilot_engaged = True
        elif m[1] == "up":
          cruise_button = CruiseButtons.RES_ACCEL
          is_openpilot_engaged = True
        elif m[1] == "cancel":
          cruise_button = CruiseButtons.CANCEL
          is_openpilot_engaged = False
        if m[1] == "leftLaneChange":
          cruise_button = CruiseButtons.Turn_left
          is_openpilot_engaged = True
          num = 50
        elif m[1] == "rightLaneChange":
          cruise_button = CruiseButtons.Turn_right
          is_openpilot_engaged = True
          num = 50
        elif m[1] == "setDitance":
          cruise_button = CruiseButtons.DistanceSet
          is_openpilot_engaged = True

      elif m[0] == "quit":
        break
      keyboard_keep_cnt = 0
    else:
      keyboard_keep_cnt += 1
    
    if keyboard_keep_cnt >= num:
      cruise_button = 0


    # print("cruise_button", cruise_button)
    # just recored data, 20ms
    if rk.frame % 2 == 0:
        t = time.time()
        vehicle_state.cruise_button = cruise_button
        dd = carlaSimulink.update(vehicle, world, world_map, vehicle_state)
        lateralActive = dd[0]
        LongitudinalActive = dd[1]
        #print('main process time is: ', time.time() - t)

    # handle lateral controller update

    if lateralActive and is_openpilot_engaged:
      steer_out = dd[2]
      steer_out = steer_rate_limit(old_steer, steer_out)
    else:
      steer_out = steer_manual * steer_manual_multiplier
    old_steer = steer_out

    # handle longitudinal controller update
    if LongitudinalActive and is_openpilot_engaged:
      accelCmd = dd[3]

      feedforward_x = [0, 5, 10, 15, 20, 25 ,30]
      feedforward_y = [0, 0.25, 0.35, 0.45, 0.5, 0.5 , 0.8]

      accelGain = 0.14 if accelCmd > 0 else 0.14
      feedforward = interp(vehicle_state.speed, feedforward_x, feedforward_y) + accelCmd*accelGain

      # TODO gas and brake is deprecated
      longitudinalCtrlCmds = accelController.update(accelCmd, vehicle_state.longtAccSensorVal, 1,  speed=vehicle_state.speed, 
                                check_saturation=True, override=False, feedforward=feedforward, deadzone=0., freeze_integrator=False)
      throttle_out = clip( longitudinalCtrlCmds, 0.0, 1.0)
      brake_out = clip(-longitudinalCtrlCmds, 0.0, 1.0)
      ##jrm add
      throttle_out = dd[7]
      if dd[4]:
        brake_out = dd[6]
      else:
        brake_out = 0

    else:
      accelController.reset()
      throttle_out = throttle_manual * throttle_manual_multiplier
      brake_out = brake_manual * brake_manual_multiplier

      old_throttle = throttle_out
      old_brake = brake_out

    # --------------Step 2-------------------------------
    steer_carla = steer_out / (max_steer_angle * STEER_RATIO * -1)
    steer_carla = np.clip(steer_carla, -1, 1)
    steer_out = steer_carla * (max_steer_angle * STEER_RATIO * -1)
    old_steer = steer_out

    
    # --------------Step 3-------------------------------
    vel = vehicle.get_velocity()
    speed = math.sqrt(vel.x**2 + vel.y**2)  # in m/s
    vehicle_state.speed = speed
    vehicle_state.vel = vel
    vehicle_state.angle = steer_out
    
    vehicle_state.is_engaged = is_openpilot_engaged
    vehicle_state.yawRate = vehicle.get_angular_velocity().z *3.14/180.
    #accelSensor = math.sqrt(vehicle.get_acceleration().y**2 + vehicle.get_acceleration().x**2)

    vehicle_state.throttlePdsPos = 0.
    vehicle_state.steerSpeed = 0.

    vehicle_state.latAccSensorVal   = imud.accelerationMeas[1]
    vehicle_state.longtAccSensorVal = imud.accelerationMeas[0]

    # --------------Step 4-------------------------------
    if rk.frame % PRINT_DECIMATION == 0:
      print("longactive: ", LongitudinalActive, "frame: ", "engaged:", is_openpilot_engaged, "; throttle: ", round(vc.throttle, 3), "; steer(c/deg): ", round(vc.steer, 3), round(steer_out, 3), "; brake: ", round(vc.brake, 3))

    if rk.frame % 5 == 0:
      vc.reverse = is_egoVehReverse
      vc.throttle = throttle_out
      vc.brake = brake_out
      vc.steer = steer_carla
      vehicle.apply_control(vc)

      t = time.time()
      world.tick()
      #print('spend time', time.time() - t)

    rk.keep_time()




  # Clean up resources in the opposite order they were created.
  imu.destroy()
  camera.destroy()
  vehicle.destroy()


def bridge_keep_alive(q: Any):
  while 1:
    try:
      bridge(q)
      break
    except RuntimeError:
      print("Restarting bridge...")


if __name__ == "__main__":
  q: Any = Queue()
  p = Process(target=bridge_keep_alive, args=(q,), daemon=True)
  p.start()

  if args.joystick:
    # start input poll for joystick
    from libs.manual_ctrl import wheel_poll_thread
    wheel_poll_thread(q)
    p.join()
  else:
    # start input poll for keyboard
    from libs.keyboard_ctrl import keyboard_poll_thread
    keyboard_poll_thread(q)

import socket
import numpy as np
import time
import math
import carla
import struct
import json
from lib.message import PubMaster, SubMaster
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from carla_lib import *


class CarlaSimulink():
  def __init__(self):
    self.wp = None
    # self.global_refer_points = global_refer_points
    # self.global_refer_points.popleft()
    self.pm = PubMaster(server_list=['carla_udp'])
    self.sm = SubMaster(server_list=['controlCommonds'] ,isDict=True)
    self.object_number = 64
    self.empty_obj = {  'id'          : 0, 
                        'dx'          : 0, 
                        'dy'          : 0, 
                        'yaw'         : 0, 
                        'velocity_x'    : 0, 
                        'velocity_y'    : 0, 
                        'distance'    : 0, 
                        'yawRate'     : 0 ,
                        'X'           : 0 ,
                        'Y'           : 0 ,
                        'Z'           : 0 ,
                        'accel_x'       : 0 ,
                        'accel_y'       : 0 ,
                        'vehicle_type': 0 ,
                        'globalX'       : 0,                 
                        'globalY'       : 0   ,
                        'leftBlinker'   : 0,                 
                        'rightBlinker'  : 0 }

    self.trackedObject = [self.empty_obj for i in range(64)]
    # udp server
    self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # udp协议
    self.server.bind(('127.0.0.1', 9090))
    self.udpAliveRC = 0
    self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
    self.frame = 0  
    self.T_trans = None     
    #plt.ion()
    #self.fig = plt.figure()


  def getObjRawData(self, vehicle, world):
    """  
    func:  get transformed objects to ego vehicle coordinates.
    params:
           -vehicle: ego vehicle class
           -word   : carla word
    return: list, 64 object attributes
    """
    raw_obstacles = get_raw_obstacles(vehicle, world)
    trans_obstacles = get_transformed_obstacles(vehicle, raw_obstacles)
    preSelectObject = get_sorted_obstacles(vehicle, trans_obstacles)

    # keep index if id has not been changed
    idDict = {}
    passiveIdxTrack = []
    idTrack = []
    newIdTrack = [obj['id'] for obj in preSelectObject]

    for i in range(self.object_number):
        if self.trackedObject[i]['id']  not in newIdTrack:
            self.trackedObject[i] = self.empty_obj
        else:
            idTrack.append(self.trackedObject[i]['id'])
        if self.trackedObject[i]['id'] !=0:
            idDict[self.trackedObject[i]['id']] = i
        else:
            passiveIdxTrack.append(i)

    cont = 0
    for obj in preSelectObject:
        if obj['id'] in idTrack:
            self.trackedObject[idDict[obj['id']]] = obj
        elif obj['id'] !=0:
            self.trackedObject[passiveIdxTrack[cont]] = obj
            cont +=0

    sorted_obstacles = self.trackedObject.copy()
    return  sorted_obstacles

  def getLaneRawData(self, vehicle, world_map):
    """ 
    brief:  
           -get transformed lane line points to ego vehicle coordinates.
    params:
           -vehicle: ego vehicle class
           -world_map: carla word map
    return: 
            list, 6 laneline list.
    """
    ### search local driving lane is visiable
    localWaypoint = world_map.get_waypoint(vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving))
    
    angle_w = localWaypoint.transform.rotation.yaw 
    angle_e = vehicle.get_transform().rotation.yaw

    if angle_w < -185:
      angle_w = angle_w + 360.

    err = angle_w - angle_e
    if abs(err) > 180:
      err = np.sign(err) * (360 - abs(err))

    #print(angle_w  ,angle_e)
    if  abs(err)  < 20.:

      # get next left lane set
      a = getNxtLneInfo(localWaypoint, self.T_trans, is_left=True)
      # get next right lane set
      b = getNxtLneInfo(localWaypoint, self.T_trans, is_left=False)
      # get current lane set
      c = getLaneCellData(localWaypoint, self.T_trans)

      # get lanset
      real_left_lane_num = len(a)
      if real_left_lane_num < 3:
        a = a + [getNullLaneData() for i in range(3 - real_left_lane_num)]
      else:
        a = a[0:3]
      real_right_lane_num = len(b)
      if real_right_lane_num < 3:
        b = b + [getNullLaneData() for i in range(3 - real_right_lane_num)]
      else:
        b = b[0:3]

      laneSet = []
      laneSet = laneSet + a + [c] + b
    else:
      laneSet = [getNullLaneData() for i in range(6)]

    return laneSet

  def create_send_msg(self, vehicle, world, world_map, vehicle_state):
    """
    brief: 
           get lane lines data and 64 objects data and vehicle state data,
           then package this  data to message bytes buffer
    params:
          -vehicle: ego vehicle class
          -world: world class
          -world_map: map class
    return:
           bytes array buffer
    """
    sorted_obstacles = self.getObjRawData(vehicle, world)
    laneSet = self.getLaneRawData(vehicle, world_map)

    #add lane data to bytes array
    send_msg = struct.pack("<B", self.udpAliveRC)

    #print(len(laneSet_send[0]))
    laneSet_send = [laneSet[3] , laneSet[0] , laneSet[4]]
    #laneSet_send = self.get_refer_line(vehicle, vehicle_state)
    for j in range(len(laneSet_send)) :
      lane = laneSet_send[j]
      fit_x = []
      fit_y = []
      num_len = len(lane['x'])
      for i in range(num_len-1):
        dy = lane['y'][i+1]- lane['y'][i]
        dx = lane['x'][i+1]- lane['x'][i]
        if lane['x'][i] > -15.1 \
          and lane['x'][i]<= 150. :
          fit_x.append(lane['x'][i])
          fit_y.append(lane['y'][i])
      # polyfit
      if len(fit_x) < 4:
        coeff = [0,0,0,0,0,0]
      else:
        # polyfit
        if len(fit_x) < 4:
          coeff = [0,0,0,0,0,0]
        else:
          """
          # three order polyfit, [c0, c1, c2, c3, x_start, x_end]
          # 1,10,20,30,40,50,60,70,80, 90, 100, 120, 150]
          k_x = [0, 1, 2, 4, 8, 10 ,15 ,20, 30]
          k_y = [1, 1, 0.8, 0.6, 0.4, 0.2, 0 ,0, 0]
          k = [interp(abs(fit_y[i]), k_x, k_y) for i in range(len(fit_x))]
          #print('xy poinrt', fit_x, fit_y)
          try:
            coeff = list(weightPolyfit(fit_x, fit_y, k)) + [0, fit_x[-1]]
          except:
            coeff = [0,0,0,0,0,0]
          X = fit_x
          Y = np.polyval([coeff[3],coeff[2],coeff[1],coeff[0]], X)
          loss = np.sum(np.multiply(fit_y-Y, fit_y-Y)) / len(fit_x)
          #print('loss',fit_x, Y - fit_y)
          """
          max_range = fit_x[-1]

          for i in range(10):
            max_range = max_range - 5
            valid_x = np.linspace(0, max_range, 20)
            valid_y = interp(valid_x, fit_x, fit_y)


            k = [1.0 for j in range(len(valid_x))]
            try:
              coeff = list(np.polyfit(valid_x, valid_y, 3)) + [0, valid_x[-1]]
            except:
              coeff = [0,0,0,0,0,0]
            Y = np.polyval([coeff[0],coeff[1],coeff[2],coeff[3]], valid_x)
            loss_avg = np.sqrt(np.sum(np.multiply(valid_y-Y, valid_y-Y)) / len(valid_x))
            loss = np.abs(valid_y-Y)
            # print(i, max_range)
            # print(i, loss_avg, '-----', valid_x, valid_y, fit_y)
            if max(loss) < 0.15:
              break

      send_msg = send_msg + struct.pack("<fffffffffffffffffffff", \
                                        lane['isAlive'], lane['width'],\
                                        lane['type'], lane['leftMarkType'], lane['leftMarkTColor'], lane['leftMarkAllowLneChg'], \
                                        lane['rightMarkType'], lane['rightMarkTColor'], lane['rightMarkAllowLneChg'], \
                                        coeff[3], coeff[2], coeff[1], coeff[0], coeff[4], coeff[5],  \
                                        0, 0, 0, 0, 0, 0)

    # add object data to bytes array
    for obj in sorted_obstacles:
        send_msg = send_msg + struct.pack("<ffffffffffff", obj['id'], obj['dx'], obj['dy'], obj['yaw'], obj['velocity_x'], \
           obj['yawRate'], obj['accel_x'], obj['X'], obj['Y'], obj['Z'], obj['velocity_y'],  obj['accel_y'] )

    # add vehice_state array
    send_msg = send_msg + struct.pack("<ffffff", vehicle_state.speed,vehicle_state.yawRate, vehicle_state.angle, 
    vehicle_state.latAccSensorVal,vehicle_state.longtAccSensorVal, vehicle_state.cruise_button,     
    )

    #print('length', len(send_msg))

    return send_msg

  def get_refer_line(self, vehicle, vehicle_state):
    #### get global refer points #########
    num_waypoint_removed = 0
    for i in range(len(self.global_refer_points)):
      print(i)
      waypoint = self.global_refer_points[i][0]
      print(waypoint)
      if len(self.global_refer_points) - num_waypoint_removed == 1:
          min_distance = 1  # Don't remove the last waypoint until very close by
      else:
          min_distance = 5 + 0.5 *vehicle_state.speed

      dist = vehicle.get_location().distance(waypoint.transform.location)
      print(dist, min_distance)
      if dist < min_distance:
          num_waypoint_removed += 1
      else:
          break
    print('remove points num', num_waypoint_removed)
    if num_waypoint_removed > 0:
        for _ in range(num_waypoint_removed):
            self.global_refer_points.popleft()

    # get lane data
    tmpX = []
    tmpY = []
    valid_num = 0
    for i in range(30):
      tmpX.append(self.global_refer_points[i][0].transform.location.x)
      tmpY.append(self.global_refer_points[i][0].transform.location.y)
      #print(self.global_refer_points[i][0].transform.location.x)
      #print(self.global_refer_points[i][0].transform.location.y)
      valid_num += 1
      #print(self.global_refer_points[i][0].transform.location)

    transformdPoints = np.dot(self.T_trans, np.stack([tmpX, tmpY, [1 for i in range(valid_num)]],axis=0)) 
    tmp_data =   getNullLaneData()
    tmp_data['x'] = list(transformdPoints[0,:]) 
    tmp_data['y'] = list(transformdPoints[1,:]) 
    tmp_data['isAlive'] = 1
    tmp_data['id'] = 0 
    tmp_data['width'] = 3.5

    #print(tmpX,tmpY)
    tt = getNullLaneData()
    tmp_data['isAlive'] = 1
    laneSet_send =[ tmp_data, tt, tt]

    return laneSet_send

  def send_prediction_log(self):
    # send tp plotjuggler
    # vs
    vs = {}
    vs['egoSpeed'] = vehicle_state.speed
    vs['egoYawrate'] = vehicle_state.yawRate
    vs['latAcc'] = vehicle_state.latAccSensorVal
    vs['longtAcc'] = vehicle_state.longtAccSensorVal
    vs['yaw'] = vehicle.get_transform().rotation.yaw*3.1415926/180.
    vs['globalX'] = vehicle.get_transform().location.x
    vs['globalY'] = vehicle.get_transform().location.y

    pm_data = {}
    pm_data['fusionObjects'] = sorted_obstacles
    pm_data['laneSet'] = laneSet
    pm_data['vehicleState'] = vs


    # send env local map for prediction dataset generation
    if self.frame % 5 ==0: 
      ### send data
      Message = json.dumps(pm_data)
      bytes_data = Message.encode('utf-8')
      #print(len(bytes_data))
      #self.udp_sock.sendto(bytes_data, ('127.0.0.1', 8228))
      #self.pm.send('carla_udp', str(pm_data),  isBytes=False)


  def update(self, vehicle, world, world_map, vehicle_state):
    '''
    main function for data exchange betwen carla and simulink
    1. send 64 objects , 4 lane lines, ego vehicle state to simulink with udp protocal.
    2. ToDO: receive steering angle and acceleration from our PnC module.
    3. ToDO: need to think about lane lines in big curve road. range end maybe below to 50m.
    '''
    ### get ego vhielce transform matrix
    dx = vehicle.get_transform().location.x
    dy = vehicle.get_transform().location.y
    dtheta = vehicle.get_transform().rotation.yaw*3.14/180

    # T_trans:transform matrix 
    self.T_trans = np.array([[np.cos(dtheta), np.sin(dtheta), -(dx*np.cos(dtheta) + dy*np.sin(dtheta))], \
                        [-np.sin(dtheta), np.cos(dtheta), dx*np.sin(dtheta) - dy*np.cos(dtheta)], \
                        [0, 0 ,1]])


    ### send carla sensor data to swc
    send_msg = self.create_send_msg(vehicle, world, world_map, vehicle_state)

    self.pm.send('carla_udp', send_msg,  isBytes=True)
    
    # update frame counter
    self.frame += 1

    # update rolling alive counter
    self.udpAliveRC +=1
    self.udpAliveRC = 0 if self.udpAliveRC > 15 else self.udpAliveRC

    ### update swc commonds data
    self.sm.update(aliveFactor = 0.1)
    #print('status', self.sm.alive['controlCommonds'])

    if self.sm.alive['controlCommonds']:
        #print(self.sm.data['controlCommonds']['counter'])
        return [self.sm.data['controlCommonds']['lateralActive'], self.sm.data['controlCommonds']['longitudinalActive'] ,\
            self.sm.data['controlCommonds']['steeringAngleCmd'], self.sm.data['controlCommonds']['accelCmd'] ,\
            ##JRM add
            self.sm.data['controlCommonds']['BrkReqActive'], self.sm.data['controlCommonds']['DrvReqActive'] ,\
            self.sm.data['controlCommonds']['BrkReqCmd'], self.sm.data['controlCommonds']['DrvReqCmd'] ,\
            ##JRM add end
         ]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0]

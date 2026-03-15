import time
import math
import numpy as np
import carla
from lib.message import PubMaster
import cv2
import scipy as sp

W, H = 1928, 1208



def int_rnd(x):
  return int(round(x))

def clip(x, lo, hi):
  return max(lo, min(hi, x))

def interp(x, xp, fp):
  N = len(xp)

  def get_interp(xv):
    hi = 0
    while hi < N and xv > xp[hi]:
      hi += 1
    low = hi - 1
    return fp[-1] if hi == N and xv > xp[low] else (
      fp[0] if hi == 0 else
      (xv - xp[low]) * (fp[hi] - fp[low]) / (xp[hi] - xp[low]) + fp[low])

  return [get_interp(v) for v in x] if hasattr(x, '__iter__') else get_interp(x)

def mean(x):
  return sum(x) / len(x)


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


# Car button codes
class CruiseButtons:
  RES_ACCEL = 4
  DECEL_SET = 3
  CANCEL = 2
  MAIN = 1
  Turn_left = 5
  Turn_right = 6
  DistanceSet = 7
  REVERSE = 8



class VehicleState:
  def __init__(self):
    self.speed = 0
    self.angle = 0
    self.bearing_deg = 0.0
    self.vel = carla.Vector3D()
    self.cruise_button = 0
    self.is_engaged = False
    self.yawRate = 0.0

    self.cruiseDistanceBtn = 0
    self.cruiseCanclBtn = 0
    self.cruiseOnBnt = 0
    self.lkaStsBtn = 0
    self.cruiseResBtn = 0
    self.cruiseSetBtn = 0
    self.latAccSensorVal = 0.
    self.longtAccSensorVal = 0.
    self.throttlePdsPos = 0.
    self.steerSpeed = 0.
    self.leftTurnLght = 0
    self.RghtTurnLght = 0



def steer_rate_limit(old, new):
  # Rate limiting to 0.5 degrees per step
  limit = 100.
  if new > old + limit:
    return old + limit
  elif new < old - limit:
    return old - limit
  else:
    return new

def accel_rate_limit(old, new):
  # Rate limiting to 0.5 degrees per step
  limit = 0.01
  if new > old + limit:
    return old + limit
  elif new < old - limit:
    return old - limit
  else:
    return new


class Camerad:
  def __init__(self, weight, height, serverName):
    self.pm = PubMaster(server_list=[serverName])
    self.width = weight
    self.height = height
    self.serverName = serverName

  def cam_callback(self, image):
    img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img = np.reshape(img, (H, W, 4))
    img = img[:, :, [0, 1, 2]].copy()
    img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
    #imgBytes = img.tobytes()

    self.pm.send(self.serverName, img, isBytes=True)


class IMUSensor():
  def __init__(self):
    self.accelerationMeas = [0,0,0]


  def imu_callback(self, imu, vehicle_state):
    self.accelerationMeas = [imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z]


def weightPolyfit(x, y, k):

  N = len(x)

  # caculate X
  for i in range(N):
    xx = np.array([[1, x[i], x[i]*x[i], x[i]*x[i]*x[i] ]]).T
    yy = y[i]

    if i==0:
      X = k[i]*np.dot(xx, xx.T)
      Y = k[i]*np.dot(xx, yy)
    else:
      X = X + k[i]*np.dot(xx, xx.T)
      Y = Y + k[i]*np.dot(xx, yy)

  return np.dot(np.linalg.inv(X + X.T), 2*Y)



def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def get_raw_obstacles(player, world):
    #get sim_world obstacles
    vehicles = world.get_actors().filter('vehicle.*')
    #print("get %d vehicles from sim_world"%len(vehicles))
    ob_dict = {}
    ob_set = []
    norm = lambda x, y, z:math.sqrt(x**2 + y**2 + z**2)

    # get min object id, make id is 0_128
    x_id = [x.id for x in vehicles]

    
    x_id.remove(player.id)
    min_id = min(x_id) if len(x_id)>0 else 0
    #print(x_id, player.id)
    for veh in vehicles:
        #print(dir(vehicles))
        t = time.time()
        a = 0#int(veh.get_light_state())

        if veh.id == player.id:
          continue
        ob_dict ={'id'          : veh.id - min_id + 1, \
                  'location'    : (veh.get_location().x, veh.get_location().y, veh.get_location().z), \
                  'theta'       : veh.get_transform().rotation.yaw, \
                  'velocity'    : (veh.get_velocity().x, veh.get_velocity().y, veh.get_velocity().z), \
                  'yawRate'     : veh.get_angular_velocity().z, \
                  'accel'       : (veh.get_acceleration().x, veh.get_acceleration().y,veh.get_acceleration().z) ,
                  'boundBox'    : (veh.bounding_box.extent.x, veh.bounding_box.extent.y ,veh.bounding_box.extent.z ) ,\
                  'vehicle_type': get_actor_display_name(veh, truncate=22),
                  'leftBlinker' :  a >> 5 & 1 ,
                  'rightBlinker' : a >> 4 & 1 ,

                  }
        
        ob_set.append(ob_dict)
            #print(veh.attributes)
    # print("get %f raw  obstalces"%len(ob_set))
    #print([x['id'] for x in ob_set])
    return ob_set

def get_transformed_obstacles(player,raw_obstacles):
    t = player.get_transform()
    t1 = player.get_location()
    v = player.get_velocity()
    
    dx = t1.x
    dy = t1.y
    dtheta = t.rotation.yaw*3.1415926/180.

    # T_trans:transform matrix 
    T_trans = np.array([[np.cos(dtheta), np.sin(dtheta), -(dx*np.cos(dtheta) + dy*np.sin(dtheta))], \
                        [-np.sin(dtheta), np.cos(dtheta), dx*np.sin(dtheta) - dy*np.cos(dtheta)], \
                        [0, 0 ,1] ])

    T_trans_v = np.array([[np.cos(dtheta), np.sin(dtheta), 0], \
                        [-np.sin(dtheta), np.cos(dtheta), 0], \
                        [0, 0 ,1] ])

    
    ob_transformed_dict = {}
    ob_transformed = []
    norm = lambda x, y:math.sqrt(x**2 + y**2)
    for ob in raw_obstacles:  

      ## translate obejct relative dx and dy
      x_abs = ob['location'][0]# - t.location.x 
      y_abs = ob['location'][1]# - t.location.y
      # ob_abs:obstacle abs tranform 
      ob_abs = np.array([x_abs, y_abs, 1])
      
      # ob_transformed_location: obstacle transformed location info
      ob_tansformed_location = np.dot(T_trans, ob_abs) 
      #print(ob_tansformed_location) 
      obj_theta = ob['theta']*np.pi/180 


      theta_abs = abs(obj_theta - dtheta)
      # make the angle error to  down to 180deg
      if theta_abs > np.pi:
          s = np.sign(math.sin(obj_theta) - math.sin(dtheta))
          relativeTheta = s*(2*np.pi - theta_abs)
      else:
          relativeTheta  = obj_theta - dtheta

      ## translate object relative vx and vy
      obj_vx = ob['velocity'][0]
      obj_vy = ob['velocity'][1]
      obj_tansformed_v = np.dot(T_trans_v, np.array([obj_vx, obj_vy, 1])) 

      ## translate object relative ax and ay
      obj_ax = ob['accel'][0]
      obj_ay = ob['accel'][1]
      obj_tansformed_a = np.dot(T_trans_v, np.array([obj_ax, obj_ay, 1])) 

      ob_transformed_dict = { 'id':ob['id'], \
                              'dx': ob_tansformed_location[0], \
                              'dy': ob_tansformed_location[1], \
                              'yaw':relativeTheta, \
                              'velocity_x': obj_tansformed_v[0], \
                              'velocity_y': obj_tansformed_v[1], \
                              'distance': norm(ob_tansformed_location[0], ob_tansformed_location[1]), \
                              'yawRate'  : ob['yawRate']*3.14/180 ,
                              'X'           :  ob['boundBox'][0] ,  
                              'Y'           :  ob['boundBox'][1] , 
                              'Z'           :  ob['boundBox'][2] , 
                              'accel_x'       :  obj_tansformed_a[0],  
                              'accel_y'       :  obj_tansformed_a[1],                
                              'vehicle_type':  ob['vehicle_type'],
                              'globalX'       : x_abs,                 
                              'globalY'       : y_abs,
                              'leftBlinker'   : ob['leftBlinker'],                 
                              'rightBlinker'  : ob['rightBlinker']   }

      ob_transformed.append(ob_transformed_dict)
    
    return ob_transformed


def get_sorted_obstacles(player, ob_transformed):
    ### delete useless object with x_dist and y_dist
    obj_temp = ob_transformed.copy()
    for obj in ob_transformed:
        if obj['dx'] > 200 or obj['dx'] < -100 or abs(obj['dy']) > 20 or abs(obj['yaw']) > 3.14/2.5:
            obj_temp.remove(obj)

    #print([(x['id'], x['dx'], x['dy']) for x in obj_temp], )
    obj_temp.sort(key=lambda x:x['distance'])
    
    ob_sorted = obj_temp

    x = [obj['dx'] for obj in ob_sorted]
    y = [obj['dy'] for obj in ob_sorted]

    #print(x, y)
    #print("get %d sorted obstacles"%len(ob_sorted))
    #print(f"nearst vehicle x_dist is {ob_sorted[0]['location'][0]:.0f},y_dist is {ob_sorted[0]['location'][1]:.0f}，yaw is {(ob_sorted[0]['yaw'])*180/3.14:.1f},vehicle_type is {ob_sorted[0]['vehicle_type']}")
    return ob_sorted


def getNxtLneInfo(localWaypoint, T_trans, is_left = True):
  # caculate total lanes
  nxtlaneWayPoints_lst = []
  nxtlaneInfoList_lst = []

  # loop
  for i in range(100): 
    if i==0 :
      lastWaypoints = localWaypoint
    else:
      lastWaypoints = nxtlaneWayPoints_lst[-1]
    if is_left:
      laneWaypoints = lastWaypoints.get_left_lane()
    else:
      laneWaypoints = lastWaypoints.get_right_lane()

    if laneWaypoints is not None and np.sign(laneWaypoints.lane_id) == np.sign(localWaypoint.lane_id):
      nxtlaneWayPoints_lst.append(laneWaypoints)

      nxtlaneInfoList_lst.append(getLaneCellData(laneWaypoints, T_trans))

    else:
      break

  return nxtlaneInfoList_lst

def getLaneCellData(lcp, T_trans):
  # get current lane data
  tmp_data = {}

  # range 
  l_sampleNumber = 25
  l_sampleDist = [-150, -110, -80, -50, -45, -40, -35, -30,-25,-20,-15,-10, \
                   1,8,12,15,20,30,50,70,80, 90, 100, 120, 150]
  
  # get lane data
  tmpX = []
  tmpY = []
  valid_num = 0

  for j in range(l_sampleNumber):
    if (l_sampleDist[j] > 0):
      nxtPoints = lcp.next(l_sampleDist[j])
    else:
      nxtPoints = lcp.previous(abs(l_sampleDist[j]))
    if len(nxtPoints) > 0:
      tmpX.append(nxtPoints[0].transform.location.x)
      tmpY.append(nxtPoints[0].transform.location.y)
      valid_num += 1
    else:
      break

  tmp_data['width'] = lcp.lane_width
  tmp_data['type'] = int(lcp.lane_type)
  tmp_data['leftMarkType'] = int(lcp.left_lane_marking.type)
  tmp_data['leftMarkTColor'] = int(lcp.left_lane_marking.color)
  tmp_data['leftMarkAllowLneChg'] = int(lcp.left_lane_marking.lane_change)
  tmp_data['rightMarkType'] = int(lcp.right_lane_marking.type)
  tmp_data['rightMarkTColor'] = int(lcp.right_lane_marking.color)
  tmp_data['rightMarkAllowLneChg'] = int(lcp.right_lane_marking.lane_change)
  
  #(valid_num)
  transformdPoints = np.dot(T_trans, np.stack([tmpX, tmpY, [1 for i in range(valid_num)]],axis=0)) 
  tmp_data['x'] = list(transformdPoints[0,:]) + [1000. for i in range(l_sampleNumber - valid_num)]
  tmp_data['y'] = list(transformdPoints[1,:]) + [0 for i in range(l_sampleNumber - valid_num)]
  tmp_data['isAlive'] = 1
  tmp_data['id'] = lcp.lane_id

  return tmp_data

def getNullLaneData():
  tmp_data = {}
  tmp_data['width'] = 0
  tmp_data['type'] = 0
  tmp_data['leftMarkType'] = 0
  tmp_data['leftMarkTColor'] = 0
  tmp_data['leftMarkAllowLneChg'] = 0
  tmp_data['rightMarkType'] = 0
  tmp_data['rightMarkTColor'] = 0
  tmp_data['rightMarkAllowLneChg'] = 0
  tmp_data['x'] =  [0 for i in range(25)]
  tmp_data['y'] =  [0 for i in range(25)]
  tmp_data['isAlive'] = 0
  tmp_data['id'] = 0   
  return   tmp_data




#!/usr/bin/python3


import socket
import pandas
import pandas as pd
import csv
import os
from datetime import datetime
from carla_visual import DataVisual
from tools.sim.message1 import PubMaster, SubMaster




sm = SubMaster(poller_list=['carla_udp'] ,isDict=True)


currentTime=datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')

if not os.path.exists(f'logger/{currentTime}'):
    os.makedirs(f'logger/{currentTime}')

f_hd1 = f'./logger/{currentTime}/train_dataset_lane.csv'
f1 = open(f_hd1, 'w', encoding='utf-8')

f_hd2 = f'./logger/{currentTime}/train_dataset_obj.csv'
f2 = open(f_hd2, 'w', encoding='utf-8')

f_hd3 = f'./logger/{currentTime}/train_dataset_ego.csv'
f3 = open(f_hd3, 'w', encoding='utf-8')

csv_writer_1 = csv.writer(f1)
csv_writer_2 = csv.writer(f2)
csv_writer_3 = csv.writer(f3)

frame = 0

#
dv = DataVisual()

# lane set
laneSet = []
objSet = []
egoSet = []

while True:
    #data, addr = sock.recvfrom(24000)
    sm.update()
    if sm.updated('carla_udp'):
        data = sm.data['carla_udp']
        ### -------------------- handle datalog ------------------------###
        msg = eval(data.decode('utf-8'))

        ### write lane data to csv.
        lane_array = []

        for tmp_data in msg['laneSet']:
            lane_array += [tmp_data['width'], tmp_data['type'] , tmp_data['leftMarkType'], \
            tmp_data['leftMarkTColor'], tmp_data['leftMarkAllowLneChg'], \
            tmp_data['rightMarkType'], tmp_data['rightMarkTColor'], \
            tmp_data['rightMarkAllowLneChg'], tmp_data['isAlive'], tmp_data['id'] ] + \
            tmp_data['x'] + tmp_data['y']
        csv_writer_1.writerow(lane_array)

        ### write object data to csv
        obj_array = []

        for tmp_data in msg['fusionObjects']:
            obj_array +=   [tmp_data['id'], tmp_data['dx'] , tmp_data['dy'], \
                            tmp_data['yaw'], tmp_data['velocity'], \
                            tmp_data['distance'], tmp_data['yawRate'], \
                            tmp_data['X'], tmp_data['Y'], tmp_data['accel'] , \
                            tmp_data['globalX'], tmp_data['globalY'], \
                            tmp_data['leftBlinker'], tmp_data['rightBlinker']
                             ] 

        csv_writer_2.writerow(obj_array)

        ### write ego car data to csv
        egoState_array = [msg['vehicleState']['egoSpeed'], msg['vehicleState']['egoYawrate'], \
                          msg['vehicleState']['latAcc'], msg['vehicleState']['longtAcc'] , \
                          msg['vehicleState']['globalX'], msg['vehicleState']['globalY'] ,  msg['vehicleState']['yaw']]

        csv_writer_3.writerow(egoState_array)

        ###-------------------- handle data visual -------------------###
        # create model inputs
        laneSet.append(lane_array )
        objSet.append(obj_array )
        egoSet.append(egoState_array )

        if len(laneSet)> 50:
            laneSet.pop(0)
            objSet.pop(0)
            egoSet.pop(0)

        if len(laneSet) >= 50:
            dv.plot_online(msg['laneSet'], msg['fusionObjects'], msg['vehicleState'], \
                            laneSet, objSet,  egoSet, False)

        frame += 1
        print(frame)

f_hd1.close()
f_hd2.close()
f_hd3.close()
#!/usr/bin/env python3
import os
import time
import re
from cffi import FFI
import numpy as np
import math
import socket
import json
from typing import Optional, List, Union, Dict


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



class PltgBrige():
    def __init__(self, port=8229):
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.port = port

    def publish(self, dict_data: Dict[str, int]):
        ### send data
        print(dict_data)
        Message = json.dumps(dict_data)
        bytes_data = Message.encode('utf-8')
        print(len(bytes_data))
        self.udp_sock.sendto(bytes_data, ('127.0.0.1', self.port))

def get_mpc_cffi():
    ffi = FFI()
    struct_def = '''
    /*=======================================================================*
     * Fixed width word size data types:                                     *
     *   int8_T, int16_T, int32_T     - signed 8, 16, or 32 bit integers     *
     *   uint8_T, uint16_T, uint32_T  - unsigned 8, 16, or 32 bit integers   *
     *   real32_T, real64_T           - 32 and 64 bit floating point numbers *
     *=======================================================================*/
    typedef signed char int8_T;
    typedef unsigned char uint8_T;
    typedef short int16_T;
    typedef unsigned short uint16_T;
    typedef int int32_T;
    typedef unsigned int uint32_T;
    typedef float real32_T;
    typedef double real64_T;

    /*===========================================================================*
     * Generic type definitions: boolean_T, char_T, byte_T, int_T, uint_T,       *
     *                           real_T, time_T, ulong_T.                        *
     *===========================================================================*/
    typedef double real_T;
    typedef double time_T;
    typedef unsigned char boolean_T;
    typedef int int_T;
    typedef unsigned int uint_T;
    typedef unsigned int ulong_T;
    typedef char char_T;
    typedef unsigned char uchar_T;
    typedef char_T byte_T;

    /* Model entry point functions */
    extern void lat_mpc_initialize(void);
    extern void lat_mpc_terminate(void);

    /* Customized model step function */
    extern void lat_mpc_step(real32_T arg_v_ego, real32_T arg_T_in[30], real32_T
      arg_y_pts[30], real32_T arg_head_pts[30], real32_T arg_k_pos, real32_T
      arg_k_heading, real32_T arg_k_control, real32_T arg_u_delay, uint8_T
      *arg_is_MPC_valid, real32_T arg_mpc_solution[30]);


    '''
    ffi.cdef(struct_def)

    return ffi


class LatMpc():
    def __init__(self):
        self.ffi = get_mpc_cffi()
        self.res    = self.ffi.new('float[30]')
        self.res_valid = self.ffi.new('unsigned char *')
        self.lib    = self.ffi.dlopen('./control/liblat_mpc.so')
        self.lib.lat_mpc_initialize()

        self.k_pos = float(0.01)
        self.k_heading = float(1.)
        self.k_control = float(500.)
        self.T_IDX = np.linspace(0.05, 2.5, 30)

        # print(self.T_IDX)

        self.last_u = 0.
        self.prev_time = 0.3
        self.des_yawrate = 0.

        self.STEER_RATIO = 16.2
        self.steer_out = 0

        self.pb = PltgBrige()



    def update(self, active, v_ego, current_yawrate_degps, traj_x,  traj_y, traj_theta, traj_t):
        # caculate y_pts and head_pts
        x_pts = np.array(self.T_IDX*v_ego)

        # first we make a polyfit
        # y_pts = list(np.polyval([C5, C4, C3, C2, C1, C0], x_pts))
        # head_pts = list(np.polyval([0, 5*C5, 4*C4, 3*C3, 2*C2, C1], x_pts))

        y_pts = np.interp(self.T_IDX, traj_t, traj_y).tolist()
        head_pts = np.interp(self.T_IDX, traj_t[0:-1], traj_theta).tolist()


        meas_curv = current_yawrate_degps*np.pi/180. / max(v_ego, 0.5)
        k_curv = interp(abs(meas_curv), [0, 0.002, 0.004, 0.006 , 0.01], [1, 1.5, 3, 4, 5])

        # caller step func
        self.lib.lat_mpc_step(v_ego, list(self.T_IDX), y_pts, head_pts, k_curv*self.k_pos, self.k_heading, self.k_control, self.last_u, self.res_valid , self.res)



        # get desired_yawrate
        prev_n = round(self.prev_time/0.05)
        desired_yawrate = (sum(list(self.res)[0:prev_n]) + self.last_u) # radps


        # yawrate to steering angle
        c_TJA_d_WheelBase_sg = 2.8
        Ffw_angD_DesPinAngle_sg = math.atan(desired_yawrate/max(0.1, v_ego) * c_TJA_d_WheelBase_sg) * 180/ np.pi  # wheel angle

        k_UndStrGrd = 2.0
        g = 9.8
        underSteerGradFct = desired_yawrate * v_ego * k_UndStrGrd /g

        ffw_wheel_angle = Ffw_angD_DesPinAngle_sg + underSteerGradFct + 0.1*(desired_yawrate*180./np.pi - current_yawrate_degps )
        #print(desired_yawrate, prev_n, ffw_wheel_angle, current_yawrate_degps * np.pi/180., Ffw_angD_DesPinAngle_sg,  underSteerGradFct)


        # get last_u
        if active:
          self.last_u = self.last_u + self.res[0]
        else:
          self.last_u = current_yawrate_degps * np.pi/180.


        # ### send data
        # pm_data = {}
        # pm_data['w_act'] = current_yawrate_degps
        # pm_data['w_des'] = desired_yawrate* 180/ np.pi
        # pm_data['pos_err'] = C0
        # pm_data['Hdang_err'] = C1

        # self.pb.publish(pm_data)

        self.steer_out =  ffw_wheel_angle

        return desired_yawrate


# lat_controller = LatMpc()
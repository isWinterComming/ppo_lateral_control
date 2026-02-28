import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import sys
import os
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import copy
from collections import deque
import torch.optim as optim
from lateral_mpc_lib.lat_mpc import LatMpc
# from Pendulum_PPO import PI_Network


class PI_Network(nn.Module):
    def __init__(self, obs_dim, action_dim, lower_bound, upper_bound) -> None:
        super().__init__()
        (
            self.lower_bound,
            self.upper_bound
        ) = (
            torch.tensor(lower_bound, dtype=torch.float32),
            torch.tensor(upper_bound, dtype=torch.float32)
        )
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, obs):
        y = F.tanh(self.fc1(obs))
        y = F.tanh(self.fc2(y))
        action =  F.tanh(self.fc3(y))

        action = (action + 1)*(self.upper_bound - self.lower_bound)/2+self.lower_bound

        return action
    
    
def warn_cosin_learningRate(epoch, warmup_epochs, optimizer, scheduler):
    if epoch < warmup_epochs:
        lr = 2e-4 * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            print(lr)
    else:
        scheduler.step()


def refercenline_line_transform(path_x, path_y, theta_bias, dx, dy) -> np.array:
    """
      Func: while the adc postion is deviated, the labeled trajectory should alse changed with new view matrix.
    """
    # need add coordinate transform
    path_x = path_x - dx
    path_y = path_y - dy
    traj_points = np.concatenate((path_x.reshape(len(path_x), 1), path_y.reshape(len(path_y), 1), np.ones((len(path_y),1))), axis=1)

    ## with comma2k dataset, gt path is rotate with yaw angle, tested!
    compensation_theta = 0.0
    transform_maxtrix = np.array([[np.cos(theta_bias+compensation_theta), np.sin(theta_bias+compensation_theta), -0],
                                  [-np.sin(theta_bias+compensation_theta), np.cos(theta_bias+compensation_theta), -0],
                                  [0,0,1]])
    # print(transform_maxtrix)
    label_traj_xyz = transform_maxtrix @ traj_points.T
    label_traj_xyz = label_traj_xyz.T

    return label_traj_xyz[:,0], label_traj_xyz[:,1]


class DrivingSimulator:
    """
    简化的驾驶模拟环境
    基于自行车模型
    """
    def __init__(self):
        # 车辆参数
        self.wheelbase = 2.7  # 轴距 (m)
        self.dt = 0.2  # 时间步长 (s)

        # 步数限制
        self.max_steps = 250
        self.current_step = 0
        self.rollout_traj_x = []
        self.rollout_traj_y = []

    def update_lane_coefficients(self, state):
        """基于更新后的车辆状态，重新计算参考线系数"""
        v_ego = state[-1]
        x_ego = state[0]
        y_ego = state[1]
        theta_ego = state[2]

        # global points to local points
        x_n, y_n = refercenline_line_transform(self.ref_x, self.ref_y, theta_ego, x_ego, y_ego)

        # interp for new sampled points
        pred_s = min(self.max_s - x_ego, 10 + v_ego * 3.0)
        sampled_x = np.linspace(0, pred_s, 20)
        sampled_y = np.interp(sampled_x, x_n, y_n)

        return np.polyfit(sampled_x, sampled_y, 3)

    def vehicle_dynamics(self, state, t, psi_rate):
        """
        自行车模型动力学
        state: [x, y, theta, v]
        kappa: 转向曲率
        """
        x, y, theta, v = state

        # 运动方程
        dx_dt = v * np.cos(theta)
        dy_dt = v * np.sin(theta)
        dtheta_dt = psi_rate
        dv_dt = 0  # 假设速度恒定，或可以添加速度控制

        return [dx_dt, dy_dt, dtheta_dt, dv_dt]

    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.rollout_traj_x = []
        self.rollout_traj_y = []
        max_ay = 2.0
        # random inputs
        v0 = max(1.0, np.random.rand() * 25.)
        # v0 = 10.0

        # max_kappa = min(max_ay / ( v0 ** 2), 0.04)
        in_A0 = (np.random.rand() - 0.5) * 2.0
        in_A1 = (np.random.rand() - 0.5) * 0.1
        in_A2 = (np.random.rand() - 0.5) * 0.004
        in_A3 = (np.random.rand() - 0.5) * 0.00001

        in_yawrate_degps =  (2 * in_A2 * v0 * 180 / np.pi + np.random.randn() * 1.0)

        # 状态变量 [x, y, theta, v]
        self.state = np.array([0.0, 0.0, 0.0, v0])  # x, y, 航向角, 速度
        self.max_s = 10. + (self.max_steps * self.dt) * 12

        # reference line 参数
        self.a0 = in_A0
        self.a1 = in_A1
        self.a2 = in_A2
        self.a3 = in_A3
        self.m_yawrate_degps = in_yawrate_degps

        self.ref_x = np.linspace(0, self.max_s, 100)
        self.ref_y = self.a0 + self.a1 * self.ref_x + self.a2 * self.ref_x * self.ref_x + self.a3 * self.ref_x *self.ref_x*self.ref_x

        desired_psi_rate_degps = in_yawrate_degps


        self.delay_time = np.random.rand() * 0.2
        # self.delay_time = 0.1
        self.delay_steps = round(self.delay_time/self.dt)
        self.control_queue = [desired_psi_rate_degps for i in range(self.delay_steps + 1)]

        self.psi_rate_degps_last = in_yawrate_degps
        self.cof_last = [self.a3, self.a2, self.a1, self.a0]

        return np.array([v0, self.a0, self.a1, self.a2 * (v0 + 5) ** 2, self.a3 * (v0 + 5) ** 3, in_yawrate_degps, self.delay_time]).reshape(1, 7), None


    def step(self, action, is_active=True):
        """
        执行一步动作
        action: yawrate radps
        """
        v_ego = self.state[-1]
        
        # print('action: ', action)
        self.control_queue.append(action[0][0])
        self.control_queue.pop(0)

        psi_rate_degps = self.control_queue[0]

        # update next state.
        t = [i*self.dt for i in range(20)]
        
        # use current control commands to get states.
        new_state = odeint(self.vehicle_dynamics, self.state, t, args=(psi_rate_degps * np.pi / 180.,))
        self.state = new_state[1]
        cof = self.update_lane_coefficients(self.state)
        self.rollout_traj_x.append(self.state[0])
        self.rollout_traj_y.append(self.state[1])
        
        # use last control commands to get states.
        # base_state = odeint(self.vehicle_dynamics, self.state, t, args=(0.0 * np.pi / 180.,))

        # measurement, add noise and get next iteration state.
        in_yawrate_degps = psi_rate_degps + 0.1 * np.random.randn()
        next_state = np.array([v_ego, cof[3], cof[2], cof[1] * (v_ego + 5) ** 2, cof[0] * (v_ego + 5) ** 3, in_yawrate_degps, self.delay_time]).reshape(1, 7)
        self.psi_rate_degps_last = psi_rate_degps
        self.cof_last = cof
        
        # according the action to rollout some steps.
        self.current_step += 1
        
        # caculate rewared.
        reward = 0.0
        preiview_step = 2
        cof_pred = self.update_lane_coefficients(new_state[preiview_step])
        # cof_base = self.update_lane_coefficients(base_state[preiview_step])
        # reward for lateral velocity decrease to zero.
        lateral_vel_pred = cof_pred[2] * (v_ego + 5)
        # lateral_vel_now = cof[2] * (v_ego + 5)
        lateral_vel_threshole = 0.5
        lat_vel_reward = 1.0 - abs(lateral_vel_pred / lateral_vel_threshole) 

        # reward for lateral position decrease to zero.
        lateral_pos_pred = cof_pred[3]
        # lateral_pos_now = cof_base[3]
        lateral_pos_threshole = 0.5
        lat_pos_reward = 1.0 - abs(lateral_pos_pred / lateral_pos_threshole) 

        state_reward = 0.2 * lat_vel_reward + 1.0 * lat_pos_reward
        reward += state_reward

        # 存活奖励
        survive_reward = 0.2
        
        # 动作惩罚（希望使用小的力）
        lateral_accel = psi_rate_degps * (v_ego + 5) * np.pi / 180.
        lateral_accel_threshold = 1.5
        lateral_accel_reward = 1.0  - abs(lateral_accel / lateral_accel_threshold)
        reward += 1.0 * survive_reward + lateral_accel_reward * 0.5
        
        # Caculate mission status.
        if abs(cof[3]) > 10.5  or abs(cof[2]) > np.pi/6.:
            mission_status = 1 # failed
            reward += -20.
        # elif abs(cof[3]) < 0.1 and abs(cof[2] * (v_ego + 5)) < 0.1:
        #     mission_status = 2 # failed
        #     reward += 20.  
        elif self.current_step >= 200:
            mission_status = 3 # trunked
        else:
            mission_status = 0
        return next_state, reward, mission_status==1, mission_status==3, None



    def test_sim(self):
        init_stat = self.reset()

        # print(init_stat.shape)
        gt_lat_mpc = LatMpc()
        gt_lat_mpc.prev_time = self.delay_time


        pos_err = []
        heading_error = []
        state_yr = []
        ctrl_state = []
        ctrl_yr = []
        for t in range(150):

            # print(init_stat)
            print('delay time: ', self.delay_time, self.control_queue)
            print('vego', init_stat[0])
            init_stat = init_stat[0]
            pos_err.append(init_stat[1])
            heading_error.append(init_stat[2] * init_stat[0])
            state_yr.append(init_stat[7])
            ctrl_yr.append(init_stat[6])
            ctrl_state.append(init_stat[5])
            gt_out_degps = (180/np.pi) * gt_lat_mpc.update(init_stat[5], init_stat[7], init_stat[0], init_stat[1], init_stat[2], init_stat[3], init_stat[4], 0.0, 0.0)
            init_stat = self.step(gt_out_degps)

        plt.plot(ctrl_yr)
        plt.plot(state_yr)
        plt.plot(pos_err)
        plt.plot(heading_error)
        plt.show()

    def test_model(self):
        model = torch.load(
        './rl_model_200.pt', map_location="cuda", weights_only=False).cuda()

        init_stat = self.reset()
        gt_lat_mpc = LatMpc()
        gt_lat_mpc.prev_time = self.delay_time

        pos_err = []
        heading_error = []
        state_yr = []
        ctrl_state = []
        ctrl_yr = []

        ctrl_state_mpc = []


        for t in range(200):

            # print(init_stat)
            print('delay time: ', self.delay_time, self.control_queue)
            init_stat = init_stat[0]
            pos_err.append(init_stat[1])
            heading_error.append(init_stat[2])
            state_yr.append(init_stat[7])
            ctrl_yr.append(init_stat[6])
            ctrl_state.append(init_stat[5])
            tx_in = torch.from_numpy(init_stat).float().cuda()
            print(tx_in.shape)
            pred_out,_,_ = model.act(tx_in.reshape(1, 9)) # here is radps
            # gt_out_degps = (180/np.pi) * gt_lat_mpc.update(init_stat[5], init_stat[7], init_stat[0], init_stat[1], init_stat[2], init_stat[3], init_stat[4], 0.0, 0.0)
            # ctrl_state_mpc.append(gt_out_degps)
            init_stat = self.step(pred_out[0].detach().cpu().numpy()[0] * 180 / np.pi)

        plt.subplot(3, 1, 1)
        plt.plot(ctrl_yr, label='model yawrate')
        plt.plot(state_yr, label='vehicle state yawrate')
        # plt.plot(ctrl_state_mpc, label='mpc yawrate')
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(pos_err, label='pose error')
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(heading_error, label='heading error')
        plt.legend()
        plt.show()

    def test_model_rl(self, n):
        model = PI_Network(7,1,-10,10)
        model.load_state_dict(torch.load('./saved_network/pi_network.pth'))
        # model.eval()
        init_stat,_ = self.reset()
        gt_lat_mpc = LatMpc()
        gt_lat_mpc.prev_time = self.delay_time

        pos_err = []
        heading_error = []
        state_yr = []
        ctrl_state = []
        ctrl_yr = []

        ctrl_state_mpc = []

        ctrl_time = [i * self.dt for i in range(100)]
        for t in range(100):
            # print(init_stat)
            # print('delay time: ', self.delay_time, self.control_queue)
            init_stat = init_stat[0]
            pos_err.append(init_stat[1])
            heading_error.append(init_stat[2])
            state_yr.append(init_stat[5])
            
            ctrl_state.append(True)
            tx_in = torch.from_numpy(init_stat).float()
            pred_out  = model(tx_in.reshape(1, 7)) # here is radps
            # print(pred_out)

            print('action mean:', pred_out.cpu().detach().numpy()  )
            gt_out_degps = (180/np.pi) * gt_lat_mpc.update(True, init_stat[3], init_stat[0], init_stat[1],  init_stat[2], init_stat[3]/(init_stat[0] + 5)**2, init_stat[4]/(init_stat[0] + 5)**3, 0.0, 0.0)
            ctrl_state_mpc.append(gt_out_degps)
            ctrl_yr.append(pred_out[0].cpu().detach().numpy() )
            init_stat ,_, _, _, _ = self.step(pred_out.detach().cpu().numpy())

        plt.subplot(4, 1, 1)
        plt.plot(ctrl_time, ctrl_yr, label='RL model yawrate')
        plt.plot(ctrl_time, state_yr, label='vehicle state yawrate')
        plt.plot(ctrl_time, ctrl_state_mpc, label='Rule-mpc yawrate')
        # plt.ylim([-10,10])
        plt.legend()
        plt.subplot(4, 1, 2)
        plt.plot(ctrl_time, pos_err, label=f'pose error:, v={init_stat[0][0]}/mps')
        plt.legend()
        plt.subplot(4, 1, 3)
        plt.plot(ctrl_time, heading_error, label=f'heading error, delay_time={self.delay_time}/s')
        plt.legend()
        plt.subplot(4, 1, 4)
        plt.plot(self.rollout_traj_x, self.rollout_traj_y, 'r', linewidth=2, label='rollout_path')
        plt.plot(self.ref_x, self.ref_y, 'b-', linewidth=2, label='target_path')
        plt.legend()
        plt.xlim([0, 200])
        plt.ylim([-40, 40])

        # plt.show()
        plt.savefig(f'./test{n}.png', dpi=150, bbox_inches='tight')
        plt.clf()
        
    # def eval_model(self, policy_model, eposide_nm):
    #     init_stat = self.reset()
    #     gt_lat_mpc = LatMpc()
    #     gt_lat_mpc.prev_time = self.delay_time

    #     pos_err = []
    #     heading_error = []
    #     state_yr = []
    #     ctrl_state = []
    #     ctrl_yr = []
    #     ctrl_state_mpc = []
    #     ctrl_time = [i * self.dt for i in range(200)]

    #     for t in range(200):
    #         init_stat = init_stat[0]
    #         # print(init_stat)
    #         # print('delay time: ', self.delay_time, self.control_queue)
    #         pos_err.append(init_stat[1])
    #         heading_error.append(init_stat[2])
    #         state_yr.append(init_stat[7])
            
    #         ctrl_state.append(init_stat[5])
    #         tx_in = torch.from_numpy(init_stat).float().cuda()
    #         pred_out, pred_std, _ = policy_model(tx_in.reshape(1, 9)) # here is radps
    #         # print(pred_out)
    #         print('action mean:', pred_out.cpu().detach().numpy(), 'action_std: ', pred_std )
    #         gt_out_degps = (180/np.pi) * gt_lat_mpc.update(True, init_stat[7], init_stat[0], init_stat[1],  init_stat[2], 0, 0, 0.0, 0.0)
    #         ctrl_state_mpc.append(gt_out_degps)
    #         ctrl_yr.append(init_stat[6])
    #         init_stat,_,_ = self.step(pred_out[0].detach().cpu().numpy()[0])
            

    #     plt.subplot(4, 1, 1)
    #     plt.plot(ctrl_time, ctrl_yr, label='RL model yawrate')
    #     plt.plot(ctrl_time, state_yr, label='vehicle state yawrate')
    #     plt.plot(ctrl_time, ctrl_state_mpc, label='Rule-mpc yawrate')
    #     # plt.ylim([-10,10])
    #     plt.legend()
    #     plt.subplot(4, 1, 2)
    #     plt.plot(ctrl_time, pos_err, label='pose error')
    #     plt.legend()
    #     plt.subplot(4, 1, 3)
    #     plt.plot(ctrl_time, heading_error, label='heading error')
    #     plt.legend()
    #     plt.subplot(4, 1, 4)
    #     plt.plot(self.rollout_traj_x, self.rollout_traj_y, 'r', linewidth=2, label='rollout_path')
    #     plt.plot(self.ref_x, self.ref_y, 'b-', linewidth=2, label='target_path')
    #     plt.legend()
    #     plt.xlim([0, 200])
    #     plt.ylim([-20, 20])

    #     # plt.show()
    #     plt.savefig(f'{eposide_nm}_eval.png')
    #     plt.cla()
    #     plt.clf()
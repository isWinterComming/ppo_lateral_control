# PilotDevTools_Simulation
## Description 
在pilot算法开发阶段,为了快速验证算法,避免频繁上车,需要一个定制的的仿真环境. 这里使用了开源的[Carla](http://carla.org/) 仿真模拟器
<img src="/home/chaoqun/Pictures/2022-03-21 18-53-23屏幕截图.png" alt="2022-03-21 18-53-23屏幕截图" style="zoom: 67%;" />

## Interface Of Carla
### Inputs
carla 需要接受整车控制信号,分别是break, steer, throttle(均归一化)

### Outputs
Carla 主要输出传感器信息和车辆动力学等信息
1. Sensor
	* Camera (主要做UI显示用, 感知本身我们不做)
	* Map (重点, 提取出车道信息, 暂时3车道, 6条车道线)
	* IMU (ax, ay, az, wx, wy, wz)
2. Vehicle
    * speed 
    * angle 
    * bearing_deg 
    * cruise_button  (复用按键信号,set, resume, disSet ...)
    * is_engaged = False
    * yawRate 
    * latAccSensorVal 
    * longtAccSensorVal 
    * throttlePdsPos 
    * steerSpeed 
    * leftTurnLght 
    * RghtTurnLght 
## How to run it.
### System requirement
1. cd到pydev_tools根目录下: pip3 install -r requirement.txt
### Install Carla from source or docker
1. cd到pydev_tools根目录下, run: sudo ./tools/sim/start_carla.sh ,系统将自动下载 carla docker
### Start docker and manager
1. 启动carla
	cd到pydev_tools根目录下, run: sudo ./tools/sim/start_carla.sh
2. 启动brige
	cd到pydev_tools根目录下, run: python3 ./tools/sim/carla_brige.py
3. 启动manager
	cd到pydev_tools根目录下, run: python3 manager.py
## How to use it.
### Keyborad Inputs
1. 激活功能, 长按或者单击数组'2', 即数字'2'复用为 pilot set/speed decrease
2. set速度设置, 长按数字'1', 速度一直+1, 短按数字'1', 速度+5, 长按数字'2', 速度一直-1, 短按数字'1', 速度-5, 
3. 转向灯, 长按数字'4', 向左变道, 长按数字'5', 向右变道, 
4. if collision has happened, you can manual control the vehicle, first you need press 'q' key, then system should enter reverse gear, if you press 'q' again, then system should enter driving gear.
  * w: throttle + 0.1 once, if you keep press 'w', system will + forever. upper bound is 1.0.
  * s: throttle - 0.1 once, if you keep press 'w', system will - forever. lower bound is -1.0.
  * a: steering angle + 5 deg once, if you keep press 'a', system will + forever. upper bound is 450deg.
  * d: steering angle - 5 deg once, if you keep press 'd', system will - forever. lower bound is -450deg.

### DataLog
1.  目前取消了一直存储数据的功能,避免磁盘空间占用过大, 取而代之,使用trigger数据recored.
2.  单击键盘空格键, 系统将自动存取前40s, 后20s的视频和数据, 目录在PydevTools根目录下 logger/realdata/
3.  如何进行数据转换以便支持数据回灌, 参阅toos/payload目录下readme.md文件
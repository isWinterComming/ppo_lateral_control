# ppo_lateral_control
ppo (rl) learning for lateral control.

## 📝 todo
- [ ] test mc & gav，优势函数对比测试蒙特卡罗采样和广义优势估计
- [ ] action增加纵向accleration，且simulator增加纵向状态，先考虑速度跟踪
- [ ] 横向planner增加状态输入，添加boundary， 参考线的点集合
- [ ] 基础参考线跟踪的reward优化以及调试，出跟踪report
- [ ] 状态量添加系统延时，车辆运动学模型同步更改
- [ ] 对比测试PPO和GRPO

## ✅ done
- [x] project init.
- [x] drl based model and envirment
- [x] tinygrad and torch version
- [x] flow match 版本的planner demo

## requirements
    pip3 install -r requirements.txt
## drl demo scripts
    python ppo_tiny.py
![Alt text](saved_images/season_reward.png)
![Alt text](saved_images/test0.png)

## flow match demo scripts
    python flow_match_tiny.py
![Alt text](saved_images/flow_match_test.png)
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from flow_mathing_traj import FlowMathingConfig, GoalFlowTrajModel

DVC = torch.device('cuda')
def create_mock_data(batch_size: int = 1) -> Dict[str, torch.Tensor]:
    """
    创建模拟输入数据
    """
    # 模拟相机特征 (batch_size, channels, height, width)
    camera_feature = torch.randn(batch_size, 512, 8, 8)
    
    # 模拟GT轨迹 (batch_size, num_points, 3) - x, y, heading
    gt_trajs = torch.randn(batch_size, 33, 3)  # 33个轨迹点，x, y, heading
    
    features = {
        'camera_feature': camera_feature,
        'gt_trajs': gt_trajs
    }

    # 目标数据为空字典
    targets = {}
    
    return features, targets


def export_to_onnx():
    from onnxconverter_common import float16
    import onnx
    """
    将模型导出为ONNX格式并转换为float16
    """
    print("开始导出ONNX模型...")
    model_path = r'./output/goalflow_traj_model.pt'
    
    # 创建示例输入
    dummy_features, _ = create_mock_data(batch_size=1)
    camera_feature: torch.Tensor = dummy_features["camera_feature"].cpu()
    gt_trajs=dummy_features['gt_trajs'].cpu()
    
    # 导入模型参数
    model = torch.load(model_path, map_location='cpu') 

    # 将模型设为评估模式
    model.cpu()
    model.eval()
    
    
    # 导出为ONNX
    torch.onnx.export(model, (camera_feature, gt_trajs), "./output/goalflow_traj_model.onnx", 
                            input_names=["features","gt_trajs"], 
                            output_names=["trajectory", "target","loss"],  
                            opset_version=15,
                            dynamo=False,
                            verbose=False)
    
    print(f"ONNX模型已保存!")
    
    # 将ONNX模型转换为float16
    onnx_model = onnx.load_model( "./output/goalflow_traj_model.onnx")
    trans_model = float16.convert_float_to_float16(onnx_model,keep_io_types=True)
    onnx.save_model(trans_model, "./output/goalflow_traj_model_f16.onnx")
    print(f"ONNX_f16模型已保存")


def test_and_save_model():
    """
    测试GoalFlowTrajModel是否能正常训练并保存模型
    """
    print("开始测试GoalFlowTrajModel...")
    
    # 配置模型参数
    config = FlowMathingConfig()
    config.training = True  # 设置为训练模式
    
    # 创建模型实例
    model = GoalFlowTrajModel(config).to(DVC)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 设置为训练模式
    model.train()
    
    print("开始训练循环...")
    
    # 进行几轮训练迭代
    num_epochs = 2  # 减少训练轮次以加快测试
    for epoch in range(num_epochs):
        try:
            # 创建模拟数据
            features, _ = create_mock_data(batch_size=1)  # 使用较小的batch size
            camera_feature: torch.Tensor = features["camera_feature"].cuda()
            gt_trajs=features['gt_trajs'].cuda()
            
            # 前向传播
            outputs = model(camera_feature, gt_trajs)
            
            # 获取损失
            loss = outputs.get('loss', None)
            
            if loss is not None:
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss not computed in model")
                break
        except Exception as e:
            print(f"发生错误: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print("训练完成，正在保存PyTorch模型...")
    
    # 保存模型
    model_path = "./output/goalflow_traj_model.pt"
    torch.save(model, model_path)
    
    print(f"PyTorch模型已保存至: {model_path}")
    
    # 导出ONNX模型
    export_to_onnx()
    
    # 测试推理模式
    print("测试推理模式...")
    config.training = False
    model.eval()
    
    with torch.no_grad():
        features, _ = create_mock_data(batch_size=1)
        camera_feature: torch.Tensor = features["camera_feature"].cuda()
        gt_trajs=features['gt_trajs'].cuda()
        outputs = model(camera_feature, gt_trajs)
        trajectory = outputs.get('trajectory', None)
        
        if trajectory is not None:
            print(f"推理成功，输出轨迹形状: {trajectory.shape}")
        else:
            print("推理未返回轨迹输出")
    
    print("所有测试完成！")


if __name__ == "__main__":
    test_and_save_model()
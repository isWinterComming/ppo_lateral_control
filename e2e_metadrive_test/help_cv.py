import numpy as np
import onnxruntime as ort
import cv2
import torch
import cv2
import numpy as np
from common.transformations.camera import get_view_frame_from_road_frame
from control.lat_mpc import LatMpc


PI = 3.1415926




class RateLimiter():
    def __init__(self):
        self.loop_time = 0.01
        self.last_y = 0
        self.limit = 1
    def update(self, y):
        if y > self.last_y + self.limit:
            return self.last_y + self.limit
        elif y < self.last_y -  self.limit:
            return self.last_y -  self.limit
        else:
            return y

class LowpassFilter():
    # y(n)=ax(n)+(1−a)y(n−1), discreted by backward difference,...
    # a=(ωc​Ts​)/(1+ωc​Ts​)

    def __init__(self,sample_freq, cutoff_freq):

        self.alpha = (2*PI*cutoff_freq/sample_freq) / (1 + 2*PI*cutoff_freq/sample_freq)
        self.last_y = 0

    def reset(self,u):
        self.last_y = u

    def update(self,u):
        y = self.alpha*u + (1 - self.alpha) * self.last_y
        self.last_y = y
        return y


class Derive():
    def __init__(self, sample_freq=10):
        self.loop_time = 1/sample_freq
        self.last_u = 0
        self.lpf = LowpassFilter(sample_freq, 1)

    def update(self, u):
        delta_u = u - self.last_u
        y = delta_u / self.loop_time

        self.last_u = u
        return self.lpf.update(y)


class MeanFilter():
    def __init__(self):
        self.n = 20
        self.queen = [0 for i in range(self.n)]

    def update(self, u):
        self.queen.pop(0)
        self.queen.append(u)
        return sum(self.queen) / self.n


MEDMDL_INSMATRIX = np.array([[910.0,  0.0,   0.5 * 512],
                                [0.0,  910.0,   47.6],
                                [0.0,  0.0,     1.0]])

# ANCHOR_TIME = np.array([0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4000000000000004, 2.6, 2.8, 3.0, 3.2, 3.5, 3.8000000000000003, 4.1, 4.4, 4.7, 5.0, 5.300000000000001, 5.6, 5.9, 6.2, 6.5])
ANCHOR_TIME = np.array([8.0 * (i/32)**2 for i in range(33)])
C2_INSMATRIX = np.array([[910.0,  0.0,   0.5 * 1164],
                        [0.0,  910.0,   0.5 * 874],
                        [0.0,  0.0,     1.0]])


def reshape_yuv(frames):
  H = (frames.shape[0]*2)//3
  W = frames.shape[1]
  in_img1 = np.zeros((6, H//2, W//2), dtype=np.uint8)

  in_img1[0] = frames[0:H:2, 0::2]
  in_img1[1] = frames[1:H:2, 0::2]
  in_img1[2] = frames[0:H:2, 1::2]
  in_img1[3] = frames[1:H:2, 1::2]
  in_img1[4] = frames[H:H+H//4].reshape((H//2, W//2))
  in_img1[5] = frames[H+H//4:H+H//2].reshape((H//2, W//2))
  return in_img1


def draw_path(device_path, img, width=0.6, height=1.22, fill_color=(255,255,255), line_color=(222,255,222)) -> None:
  bs = device_path.shape[0]
  device_path_l = device_path + np.array([0, 0, height])                                                                    
  device_path_r = device_path + np.array([0, 0, height]) 

  # calib frame to raod frame                                                         
  device_path_l[:,1] -= width                                                                                               
  device_path_r[:,1] += width
  device_path_l[:,2] = -1*device_path_l[:,2]  
  device_path_l[:,1] = -1*device_path_l[:,1] 
  device_path_r[:,2] = -1*device_path_r[:,2]  
  device_path_r[:,1] = -1*device_path_r[:,1] 

  m1 = get_view_frame_from_road_frame(0, 0, 0, 0,0)
  calib_pts = np.vstack((device_path_l.T, np.ones((1,bs)) ))
  view_pts = m1 @ calib_pts
  for i in range(bs):
    view_pts[0,i] = view_pts[0,i]/max(view_pts[2,i], 2)
    view_pts[1,i] = view_pts[1,i]/max(view_pts[2,i], 2)
    view_pts[2,i] = view_pts[2,i]/max(view_pts[2,i], 2)

  img_pts_l = MEDMDL_INSMATRIX @ view_pts
  img_pts_l = img_pts_l.astype(int)
  calib_pts = np.vstack((device_path_r.T, np.ones((1,bs)) ))
  view_pts = m1 @ calib_pts
  for i in range(bs):
    view_pts[0,i] = view_pts[0,i]/max(view_pts[2,i], 2)
    view_pts[1,i] = view_pts[1,i]/max(view_pts[2,i], 2)
    view_pts[2,i] = view_pts[2,i]/max(view_pts[2,i], 2)

  img_pts_r = MEDMDL_INSMATRIX @ view_pts
  img_pts_r = img_pts_r.astype(int)
  for i in range(1, img_pts_l.shape[1]):
    #check valid
    if img_pts_l[2,i] >0 and img_pts_r[2,i] >0:
      u1 = img_pts_l[0, i-1]
      v1 = img_pts_l[1, i-1]
      u2 = img_pts_r[0, i-1]
      v2 = img_pts_r[1, i-1]
      u3 = img_pts_l[0, i]
      v3 = img_pts_l[1, i]
      u4 = img_pts_r[0, i]
      v4 = img_pts_r[1, i]
      pts = np.array([[u1,v1],[u2,v2],[u4,v4],[u3,v3]], np.int32).reshape((-1,1,2))
      cv2.fillPoly(img,[pts],fill_color)
      cv2.polylines(img,[pts],True,line_color)



def get_calib_matrix(cam_insmatrixs=C2_INSMATRIX, pos_bias=0, theta_bias=0, 
                     ang_x=0, ang_y=0, ang_z=0, dev_height=1.22, lat_bias=0) -> np.array:
	"""
		Func: if the camera heading angle of position has been changed, the trasform matrix is changed too.
	"""
	camera_frame_from_ground = np.dot(cam_insmatrixs,
																			get_view_frame_from_road_frame(ang_x, ang_y, ang_z, dev_height, lat_bias))[:, (0, 1, 3)]
	calib_frame_from_ground = np.dot(MEDMDL_INSMATRIX,
																			get_view_frame_from_road_frame(0, 0, 0, 1.22))[:, (0, 1, 3)]
	calib_msg = np.dot(camera_frame_from_ground, np.linalg.inv(calib_frame_from_ground))

	return calib_msg
  

## onnx planning demo model runner, get images, pred trajectory!
class PlanModel():
    def __init__(self, cuda=False):
        options = ort.SessionOptions()
        provider = 'CUDAExecutionProvider' if cuda else 'CPUExecutionProvider'
        self.session = ort.InferenceSession(f'./planning_model_f16.onnx', options, [provider])

        # print shapes
        input_shapes = {i.name: i.shape for i in self.session.get_inputs()}
        output_shapes = {i.name: i.shape for i in self.session.get_outputs()}
        print('input shapes : ', input_shapes)
        print('output shapes: ', output_shapes)

        self.feat_buff = np.zeros((1, 20, 512))
        self.last_img_latent = np.zeros((3, 128, 256))
        self.mpc_controller = LatMpc()

	#staticmethod
    def softmax_2d(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

      
    def run(self, img_bgr):
        img = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (256, 128), interpolation=cv2.INTER_LINEAR)
        feed_img = img.transpose(2,0,1) 

        big_imgs = np.concatenate([self.last_img_latent[None,:], feed_img[None,:]], axis=1)

        # Trainging started with the 2rd image.
        out_preds, latent_mem = self.session.run(None, {'big_imgs_last': big_imgs.astype(np.float16), 'hiddenst_in': self.feat_buff.astype(np.float16)})

        # Time series stitch
        self.feat_buff = np.roll(self.feat_buff, shift=-1, axis=1)
        self.feat_buff[:, -1, :] = latent_mem
        self.last_img_latent = feed_img

        preds_buffer = out_preds[:, 512:]
        pred_cls_obj = preds_buffer[:,0:5]
        pred_traj_obj = preds_buffer[:,5:5+5*33*3].reshape(5, 33, 3)
        pred_traj_obj_std = preds_buffer[:,5+5*33*3:5+5*33*3*2].reshape(5, 33, 3)
        print("predict velocity is : ",  (preds_buffer[:,5+5*33*6]))
        
        pred_index = np.argmax(self.softmax_2d(pred_cls_obj), axis=1)
        best_traj_np = pred_traj_obj[pred_index]
        print(best_traj_np.shape, pred_index)
        return best_traj_np, preds_buffer[:,5+5*33*6]


class PlanModelV3():
    def __init__(self, cuda=False):
        self.devc = torch.device('mps')
        devc = torch.device('mps')

        # from models.e2e_model import PlanningModel
        # plan_model = PlanningModel()
        # plan_model.load_state_dict(torch.load('./planning_model_v3.pt', map_location=devc, weights_only=True))
        plan_model = torch.load('./planning_model.pt', map_location=devc, weights_only=False).to(devc)    # 默认GPU
        plan_model.eval()

        self.policy_net = plan_model
        self.feat_buff = torch.zeros((1, 20, 512)).to(self.devc)
        self.last_img_latent = torch.zeros((1, 3, 128, 256)).to(self.devc)
        self.mpc_controller = LatMpc()
        
    def run(self, img_bgr):
        img = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (256, 128), interpolation=cv2.INTER_LINEAR)
        feed_img = img.transpose(2,0,1) 
        with torch.no_grad():
            input_bev = torch.from_numpy((feed_img[None,:])).to(self.devc)
            latent_feature = torch.cat([self.last_img_latent, input_bev], dim=1)

            # Trainging started with the 2rd image.
            out_preds, latent_mem = self.policy_net(latent_feature, self.feat_buff)

            # Time series stitch
            self.feat_buff = self.feat_buff.roll(shifts=-1, dims=1)
            self.feat_buff[:, -1, :] = latent_mem
            self.last_img_latent = input_bev.clone()

            preds_buffer = out_preds[:, 512:]
            pred_cls_obj = preds_buffer[:,0:5]
            pred_traj_obj = preds_buffer[:,5:5+5*33*3].reshape(5, 33, 3)
            pred_traj_obj_std = preds_buffer[:,5+5*33*3:5+5*33*3*2].reshape(5, 33, 3)
            elu = torch.nn.ELU()
            print("predict velocity is : ",  (preds_buffer[:,5+5*33*6]),  elu(preds_buffer[:,5+5*33*6 + 6]) + 1.0)
            
            pred_index = pred_cls_obj.softmax(dim=1).argmax(dim=1)
            best_traj_np = pred_traj_obj[pred_index].cpu().detach().numpy()
            print(best_traj_np.shape, pred_index)
            return best_traj_np
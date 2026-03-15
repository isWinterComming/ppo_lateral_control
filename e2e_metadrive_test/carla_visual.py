import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import time
import torch
import torch.nn as nn
import sys

#from tools.prediction.predict_lstm import Predictor


class DataVisual():
	def __init__(self,):
		plt.ion()
		self.fig = plt.figure()

		#self.loss_func = nn.MSELoss()
		#self.predictor = Predictor()
		#self.predictor.load(path='./tools/prediction/')

	def get_one_input(self, laneSet, objSet, egoInfo):
		#print(index)
		data_in_lane = np.array(laneSet)
		data_in_obj  = np.array(objSet)
		data_in_ego  = np.array(egoInfo)  
		print(data_in_lane.shape, data_in_ego.shape)

		### process object data
		#print(data_in_obj.shape)
		data_in_obj = data_in_obj[:, 0:32*12]  # only use 32 objects, original is 64 

		### process ego data
		data_in_ego = np.delete(data_in_ego, [4, 5, 6], axis=1) # delete location as a inputs.

		# delete global X
		del_idx = [(12*i + 10 ) for i in range(32)] + [(12*i + 11 ) for i in range(32)] 
		data_in_obj = np.delete(data_in_obj, del_idx, axis=1) # delete location as a inputs.


		dd = np.column_stack((data_in_ego,  data_in_obj, data_in_lane))
		return torch.Tensor(dd)

	def plot_offline(self):
		csv_data = pd.read_csv('./logger/2022-04-13-08-44-51/train_dataset_lane.csv', index_col=0).to_dict('split')

		idx = np.array(csv_data['index'])
		data = np.array(csv_data['data'])


		dd = np.hstack(( idx.reshape(idx.shape[0], 1),  data) )


		csv_data1 = pd.read_csv('./logger/2022-04-13-08-44-51/train_dataset_obj.csv', index_col=0).to_dict('split')

		idx1 = np.array(csv_data1['index'])
		data1 = np.array(csv_data1['data'])


		dd1 = np.hstack(( idx1.reshape(idx1.shape[0], 1),  data1) )


		print(dd)



		for i in range(dd.shape[0]):
			st = time.time()
			ax = fig.add_subplot(111)
			plt.title("Object Trajectory Prediction") 
			plt.xlabel("x axis caption") 
			plt.ylabel("y axis caption") 

			plt.xlim(-40, 40)
			plt.ylim(-100, 200)

			### draw lane lines
			for j in range(6):
				l_width = dd[i, 40*j+0]
				X = dd[i, 40*j+10:40*j+25]
				Y = dd[i, 40*j+25:40*j+40]
				ax.plot(Y, X,  'g--')
				#ax.plot(Y+l_width/2., X,  'g-')

			# draw objects

			for j in range(32):
				dx = float(dd1[i, 11*j+1])
				dy = float(dd1[i, 11*j+2])
				X  = float(dd1[i, 11*j+7])
				Y  = float(dd1[i, 11*j+8])

				yaw = float(dd1[i, 11*j+3])

				print(dx, dy, X, Y)
				r2 = patches.Rectangle((dy - Y, dx - X), 2*Y, 2*X, color="blue",  alpha=0.50)
				t2 = mpl.transforms.Affine2D().rotate_deg(-yaw*180/3.14) + ax.transData
				r2.set_transform(t2)
				ax.add_patch(r2)

			r2 = patches.Rectangle((0 - 1.4, 0 - 2.5), 2.8, 5.0, color="red",  alpha=0.50)
			ax.add_patch(r2)

			plt.plot()
			plt.pause(0.0001)
			plt.clf()
			print(time.time() - st)

	def plot_online(self, laneSet, objSet, egoInfo, lane_sets, obj_sets, ego_sets, isPredict=False):
		st = time.time()
		ax = self.fig.add_subplot(111)
		plt.title("Object Trajectory Prediction") 
		plt.xlabel("x axis caption") 
		plt.ylabel("y axis caption") 

		plt.xlim(-40, 40)
		plt.ylim(-100, 200)

		### draw lane lines
		for lane in laneSet:
			l_width = lane['width']
			X = np.array(lane['x'])
			Y = np.array(lane['y'])
			#ax.plot(Y, X,  'g--')
			ax.plot(Y+l_width/2., X,  'b-')
			ax.plot(Y-l_width/2., X,  'b-')

		# draw objects
		for obj in objSet:
			dx = float(obj['dx'])
			dy = float(obj['dy'])
			X  = float(obj['X'])
			Y  = float(obj['Y'])
			yaw = float(obj['yaw'])

			#print(dx, dy, X, Y)
			r2 = patches.Rectangle((0, 0), 2*Y, 2*X, color="blue",  alpha=0.50)
			t2 = mpl.transforms.Affine2D().rotate_deg(-yaw*180/3.14).translate(dy-Y, dx-X) + ax.transData
			r2.set_transform(t2)
			ax.add_patch(r2)

		#r2 = patches.Rectangle((0 - 1.4, 0 - 2.5), 2.8, 5.0, color="red",  alpha=0.50)
		#ax.add_patch(r2)

		if isPredict:
			# draw predicted trajectory
			tx = self.get_one_input(lane_sets, obj_sets, ego_sets)

			input_data = torch.unsqueeze(tx, dim=0)
			#print(input_data.shape)
			output = self.predictor.update(input_data.to(torch.device('cuda:0')))
			outputs = torch.squeeze(output)#[0]

			outputs = outputs.cpu().detach().numpy()
			isValid = outputs[0:32]
			outputs = outputs[32:3232]
			print(isValid)
			for i in range(32):
				if isValid[i] > 0.5:
					X =   np.multiply(outputs[100*i      : 100*i + 50] , [1 for i in range(50)])
					Y =   np.multiply(outputs[100*i + 50 : 100*i + 100]  , [1 for i in range(50)])
					#print(X, Y)
					dtheta = -1*torch.squeeze(tx)[-1, 4 + i*10 + 3].cpu().detach().numpy()
					dx     = -1*torch.squeeze(tx)[-1, 4 + i*10 + 1].cpu().detach().numpy()
					dy     = -1*torch.squeeze(tx)[-1, 4 + i*10 + 2].cpu().detach().numpy()
					T = np.array([[np.cos(dtheta), np.sin(dtheta), -(dx)], \
					              [-np.sin(dtheta), np.cos(dtheta),- dy ], \
					              [0, 0 ,1] ])
					xx = np.dot(T, np.stack([X, Y, [1 for i in range(50)]],axis=0))
					plt.plot(xx[1,:], xx[0,:],  'r--')
		plt.plot()
		plt.pause(0.0001)
		plt.clf()
		print( time.time() - st)
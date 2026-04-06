  def draw_bev_map(self, tx, ty, fig, bev_img_data):
    '''
    Func: draw ground truth object trajectory and lanelines
    param: 
          1. model_in, size is (50,216)
          2. grounth label, size is (1, )
    '''
    # config figure axix ax
    model_in = torch.squeeze(tx).cpu().detach().numpy()
    gt_traj_raw = torch.squeeze(ty).cpu().detach().numpy()
    ax = fig.add_subplot(111)
    plt.xlim((-15, 15))
    plt.ylim((-40, 100))
    plt.gca().invert_xaxis()


    obj_len = 8
    ego_len  = 6
    lines_len = 20

    ego_traj = gt_traj_raw[0:PRED_TRAJ_LEN]
    obj_traj = gt_traj_raw[PRED_TRAJ_LEN:]
    obj_traj_valid = gt_traj_raw[-1*OBJECT_NUM:]
    obj_input = model_in[-1,ego_len+lines_len:]



    laneLines_in = model_in[-1, ego_len:ego_len+lines_len]



    print(tx.shape, bev_img_data.shape)
    pred_cls, pred_traj = self.pred_model(tx, bev_img_data)


    # draw ground truth object trajectory
    for i in range(32):

      if obj_traj_valid[i]:
        # deal with raw
        obj_x = obj_input[obj_len*i + 1]*100
        obj_y = obj_input[obj_len*i + 2]*10

        dx = obj_x
        dy = obj_y

        X  = 5
        Y  = 1.4
        yaw = 0.0

        # r2 = patches.Rectangle((0, 0), 2*Y, 2*X, color="blue",  alpha=0.50)
        # t2 = mpl.transforms.Affine2D().rotate_deg(-yaw*180/3.14).translate(dy-Y, dx-X) + ax.transData
        # r2.set_transform(t2)
        # ax.add_patch(r2)

        x = obj_traj[i*48:i*48+24]    + obj_x
        y = obj_traj[i*48+24:i*48+48] + obj_y
        ax.plot(y, x, 'o', color="black",)

        # deal with predict
        model_gt_valid = ty[0, 48+48*32+i]
        model_gt_traj = ty[0, 48+48*i:48+48*i+48]
        model_input = tx


        pred_cls_obj  = pred_cls[:,i,:]
        pred_traj_obj = pred_traj[:,i,:,:,:]



        traj_cls = torch.squeeze(pred_cls_obj)
        traj_pred = torch.squeeze(pred_traj_obj)

        # get predict info
        traj_prob = self.prob_func(traj_cls).cpu().detach().numpy()

        # plot traj
        best_traj_x = np.array([0 for i in range(24)], dtype="float64")
        best_traj_y = np.array([0 for i in range(24)], dtype="float64")

        best_traj = traj_pred[np.argmax(traj_prob), :, :]

        gt_traj_pos   = torch.cat((model_gt_traj[0:24].view(1,24,1), \
                                   model_gt_traj[24:48].view(1,24,1)), dim=2) # label size: [B, M, 2]

        reg_loss = self.reg_loss_func_L1(best_traj.view(1,24,2), gt_traj_pos).mean()

        print(reg_loss)

        for j in range(self.traj_num):
          traj_x  = traj_pred[j, :, 0].cpu().detach().numpy()
          traj_y  = traj_pred[j, :, 1].cpu().detach().numpy()
          best_traj_x += traj_prob[j]*traj_x
          best_traj_y += traj_prob[j]*traj_y

          # p = round(traj_prob[j], 2)
          print('prob', traj_prob[j])
          ax.plot(traj_y + obj_y, traj_x + obj_x, 'o-', color='blue',\
                    linewidth='5',label='prob-{:.2f}'.format(traj_prob[j]), alpha=float(traj_prob[j]))
        # ax.plot(best_traj_y + obj_y, best_traj_x + obj_x, 'o-', color='red',\
        #              linewidth='5', alpha=float(1.0))

    # draw lane lines  
    for k in range(4):
      line_range = laneLines_in[5*k+0] * 80
      A0   =  laneLines_in[5*k+1] * 6
      A1   =  laneLines_in[5*k+2] * 0.1
      A2   =  laneLines_in[5*k+3] * 0.01
      A3   =  laneLines_in[5*k+4] * 0.0001
      X =  np.linspace(0, line_range, 20)
      Y =  np.polyval([A3, A2, A1, A0], X) 
      ax.plot(Y, X, 'b')
    plt.plot()
    plt.pause(0.1)
    plt.clf()
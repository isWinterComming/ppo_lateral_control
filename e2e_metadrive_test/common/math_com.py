import numpy as np


TRAJ_T_IDX = [0.0, 0.009765625, 0.0390625, 0.087890625, 0.15625, 0.244140625, 0.3515625, 0.478515625, 0.625, 0.791015625, 0.9765625, 1.181640625, 1.40625, 1.650390625, 1.9140625, 2.197265625, 2.5, 2.822265625, 3.1640625, 3.525390625, 3.90625, 4.306640625, 4.7265625, 5.0]
#TRAJ_T_IDX = [0.2*k for k in range(24)]




def laneline_points_calc(linepara):
  sample_n = 20
  x_p = np.linspace(0, linepara[:,0], sample_n)
  coeff = linepara[:,1:5]
  coeff = coeff[:, ::-1]

  y_p = np.polyval(coeff.T, x_p)

  out = np.zeros((linepara.shape[0], sample_n*2))
  out[:,0:sample_n] = x_p.T
  out[:,sample_n:sample_n*2] = y_p.T

  return out


def coord_translate(dx, dy, dpsi, x, y):
    x_t = x*np.cos(dpsi) + y*np.sin(dpsi) - dx
    y_t = -1*x*np.sin(dpsi) + y*np.cos(dpsi) - dy

    return x_t, y_t
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

 
def get_line_points( line_para):
  line_0_x = np.linspace(0, line_para[0], 20)
  line_coeff = line_para[1:5]
  line_coeff.reverse()
  line_0_y = np.polyval(line_coeff, line_0_x) 
  return list(line_0_x) + list(line_0_y)

def trajectory_calc( vEgo, wEgo, n, dt):
  traj_points_gt = np.zeros((n,2))
 
  for i in range(n):
      dx = -1*vEgo[i]*dt
      rho = wEgo[i]/max(vEgo[i], 0.1)
      dy = -1*0.5*rho*dx*dx
      dpsi = -1*wEgo[i]*dt

      xx = traj_points_gt[n-i:n,0]
      yy = traj_points_gt[n-i:n,1]

      x_t, y_t = coord_translate(dx, dy, dpsi, xx, yy)
      traj_points_gt[n-i:n,0] = x_t
      traj_points_gt[n-i:n,1] = y_t
      #print(tmp.shape, traj_points.shape)
      #traj_points = np.concatenate([np.array([[0,0]]), tmp], axis=0)

  return np.concatenate([np.array([[0,0]]), traj_points_gt], axis=0)

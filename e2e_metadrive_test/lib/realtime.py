import time
import multiprocessing
import os 

def sec_since_boot():
  return time.time()

def set_process_cores(core_id):
  pid = os.getpid()
  os.sched_setaffinity(pid, core_id)
  print("CPU affinity mask is modified for process id % s" % pid) 


class Ratekeeper():
  def __init__(self, rate, rk_name="default", print_delay_threshold=0.01):
    """Rate in Hz for ratekeeping. print_delay_threshold must be nonnegative."""
    self._interval = 1. / rate
    self._next_frame_time = sec_since_boot() + self._interval
    self._print_delay_threshold = print_delay_threshold
    self._frame = 0
    self._remaining = 0
    self._process_name = multiprocessing.current_process().name
    self._rk_name = rk_name
    self.current_time = sec_since_boot()
    self._consume_time = 0

  @property
  def frame(self):
    return self._frame

  @property
  def remaining(self):
    return self._remaining
  
  def get_timestamp(self):
    # return int(time.mktime(time.localtime(time.time())))
    return int(time.time()*1000)

  # Maintain loop rate by calling this at the end of each loop
  def keep_time(self):
    lagged = self.monitor_time()
    if self._remaining > 0:
      time.sleep(self._remaining)
    self.current_time = sec_since_boot()
    return lagged

  # this only monitor the cumulative lag, but does not enforce a rate
  def monitor_time(self):
    lagged = False
    # remaining = self._next_frame_time - sec_since_boot()
    self._consume_time = sec_since_boot() - self.current_time
    self._remaining = self._interval - self._consume_time
    # self._next_frame_time += self._interval
    if self._print_delay_threshold is not None and self._remaining < -self._print_delay_threshold:
      print("%s lagging by %.2f ms" % (self._rk_name, -self._remaining * 1000))
      lagged = True
    self._frame += 1    
    return lagged


import zmq
import threading
import time
import multiprocessing
from RtNode.value import REMOTE_HOST_MODE, REMOTE_HOST_IP

### server list 
server_dict = {'ch_can':   (8201, 50, 1, 0),
               'pv_can':   (8202, 50, 1, 0),
               'send_can': (8203, 50, 1, 0),
               'control':  (8204, 50, 0, 0),
               'cam':      (8228, 20, 0, 0),
               'cam1':      (9228, 20, 0, 0),
               'modeld':      (9229, 20, 1, 0),

               'rear_cam':      (8229, 20, 0, 0),
               'calibrate_request': (8301, 0, 1, 0),
               'read_request': (8302, 50, 1, 0),
               'calibrate_response': (8303, 0, 1, 0),
               'log_request': (8304, 0, 1, 0),
               'measurement': (8305, 0, 1, 0),
               'interal_signals': (8306, 50, 1, 0),
               'carla_udp': (8308, 10, 0, 0),
               'controlCommonds': (8309, 50, 1, 0),
               'adptrin': (8310, 100, 0, REMOTE_HOST_MODE),
               'adptrout': (8311, 100, 0, REMOTE_HOST_MODE),
               'em': (8312, 20, 0, REMOTE_HOST_MODE),
               'fct': (8313, 50, 0, REMOTE_HOST_MODE),
               'pln': (8314, 10, 0, REMOTE_HOST_MODE),
               'pln_new': (8399, 10, 0, REMOTE_HOST_MODE),
               'ctrl': (8315, 100, 0, REMOTE_HOST_MODE),
               'calibrate_response_adptrin': (8316, 0, 1, 0),
               'calibrate_response_adptrout': (8317, 0, 1, 0),
               'calibrate_response_em': (8318, 0, 1, 0),
               'calibrate_response_fct': (8319, 0, 1, 0),
               'calibrate_response_pln': (8320, 0, 1, 0),
               'calibrate_response_ctrl': (8321, 0, 1, 0),
               'em_m': (8322, 20, 0, REMOTE_HOST_MODE),
               'fct_m': (8404, 50, 0, REMOTE_HOST_MODE),
               'pln_m': (8405, 10, 0, REMOTE_HOST_MODE),
               'ctrl_m': (8406, 100, 0, REMOTE_HOST_MODE),
               'ui_display': (8326, 25, 0, 0),
               'ld_cam':(8327, 20, 0, 0),
               'vehicle': (8401, 50, 0, REMOTE_HOST_MODE),
               'lane': (8402, 50, 0, REMOTE_HOST_MODE),
               'od': (8411, 50, 0, REMOTE_HOST_MODE),

               'pred': (8403, 20, 0, REMOTE_HOST_MODE),
               'systemd': (8410, 150, 0, 0),
               'ehr': (8407, 10, 0, REMOTE_HOST_MODE),
               'geo': (8408, 10, 0, REMOTE_HOST_MODE),
               'ehp': (8409, 50, 0, REMOTE_HOST_MODE),
               'fct_temp': (8412, 50, 0, 0),
               'ehp': (8409, 50, 0, REMOTE_HOST_MODE),
               'traffic': (8450, 50, 0, REMOTE_HOST_MODE)  ,
               'metadrive': (9208, 20, 1, 0),            
}




def sec_since_boot():
  return time.time()



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

  @property
  def frame(self):
    return self._frame

  @property
  def remaining(self):
    return self._remaining

  # Maintain loop rate by calling this at the end of each loop
  def keep_time(self):
    lagged = self.monitor_time()
    if self._remaining > 0:
      time.sleep(self._remaining)
    return lagged

  # this only monitor the cumulative lag, but does not enforce a rate
  def monitor_time(self):
    lagged = False
    remaining = self._next_frame_time - sec_since_boot()
    self._next_frame_time += self._interval
    if self._print_delay_threshold is not None and remaining < -self._print_delay_threshold:
      print("%s lagging by %.2f ms" % (self._rk_name, -remaining * 1000))
      lagged = True
    self._frame += 1
    self._remaining = remaining
    return lagged




def new_message(service=None, size=None):
  dat = log.Event.new_message()
  dat.logMonoTime = time.time()
  dat.valid = True
  if service is not None:
    if size is None:
      dat.init(service)
    else:
      dat.init(service, size)
  return dat


def create_socket_pub(port=5555):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:" + str(port))   #多个客户端连接同样的地址
    return socket

def create_socket_sub(port=5555, ip="localhost"):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE,''.encode('utf-8'))  # 接收所有消息
    socket.setsockopt(zmq.CONFLATE, True)
    socket.connect(f"tcp://{ip}:{port}")   #多个客户端连接同样的地址

    return socket



class zmq_sub_thread(threading.Thread):
    def __init__(self, port=8001, isDict=False):
        # 重写threading.Thread的__init__方法时，确保在所有操作之前先调用threading.Thread.__init__方法
        super().__init__()
        self.socket = create_socket_sub(port=port)
        self.data = None
        self._running = True
        self.isDict = isDict
        self.rolllingCount = 0
        self.valid = False

    def run(self):
        while self._running:
            data = self.socket.recv()

            if self.isDict:
                self.data = eval(data.decode('utf-8'))
            else:
                self.data = data
            self.rolllingCount +=1
            if self.rolllingCount >15 :
                self.rolllingCount = 0

    def terminate(self):
        self._running = False


class SubMaster():
  def __init__(self, server_list=[], poller_list=[], isDict=False):
    self.socks = {}
    self.threads = {}
    self.data = {}
    self.recvTime = {}
    self.recvDeltaTm = {}
    self.alive = {}
    self.freq = {}
    self.avgHistoryLen = 20
    #self.updated = {}
    self.ignore_alive = {}
    self.m_freq = {}

    self.poller = zmq.Poller()
    self.server_list = server_list
    self.poller_list = poller_list

    #backend threads for non_block server
    for server in server_list:
        if server_dict[server][3]:
          ip = REMOTE_HOST_IP
        else:
          ip = "localhost"
        self.threads[server] = create_socket_sub(port=server_dict[server][0], ip=ip)
        self.data[server] = None
        self.recvTime[server] = 0
        self.recvDeltaTm[server] = [1 for i in range(self.avgHistoryLen)]
        self.freq[server] = server_dict[server][1]
        self.alive[server] = False

    #poller blocking server
    for server in poller_list:
        if server_dict[server][3]:
          ip = REMOTE_HOST_IP
        else:
          ip = "localhost"
        self.socks[server] = create_socket_sub(port=server_dict[server][0], ip=ip)
        self.poller.register(self.socks[server], zmq.POLLIN)
        self.data[server] = None

    self.ss = None

  def __getitem__(self, s: str):
    return self.data[s]


  def all_alive(self, service_list=None):
    if service_list is None:  # check all
      service_list = self.alive.keys()
    return all(self.alive[s] for s in service_list )


  def stop(self):
      for key, value in self.threads:
          value.terminate()

  def update(self, timeout=1, isDict=True, aliveFactor = 0.1):
      currentTime = sec_since_boot()
      if self.poller_list:
          self.ss = dict(self.poller.poll(timeout=timeout))
      for s, sock in self.threads.items():

          # caculate message 
          try:
              data = sock.recv(zmq.DONTWAIT)
          except:
              data = None

          if data is not None:
              # assign data to the submaster 
              if server_dict[s][2] == 1:  # python-dict formar message
                  self.data[s] = eval(data.decode('utf-8'))
              else:
                  self.data[s] = data
              # assign time
              if self.recvTime[s] > 0.00001:
                  self.recvDeltaTm[s].append(currentTime - self.recvTime[s])
                  self.recvDeltaTm[s].pop(0)
              self.recvTime[s] = currentTime

          # caculate alive status
          #print(time.time() - currentTime)
          if self.freq[s] > 0.01:
            # alive if delay is within 10x the expected frequency
            #print((currentTime - self.recvTime[s]))
            self.alive[s] = (currentTime - self.recvTime[s]) < (10. / self.freq[s])

            # alive if average frequency is higher than 90% of expected frequency
            avg_dt = sum(self.recvDeltaTm[s]) / self.avgHistoryLen

            self.m_freq[s] = 1/max(avg_dt, 0.0001)
            expected_dt = 1 / (self.freq[s] * aliveFactor)
            self.alive[s] = self.alive[s] and (avg_dt < expected_dt)
            #print(self.freq[s] * 0.90, expected_dt, avg_dt)
          else:
              self.alive[s] = True

  def updated(self, server):

      if self.ss.get(self.socks[server]) == zmq.POLLIN:
          message = self.socks[server].recv()

          self.data[server] = message
          
          return True
      else:
          return False





class PubMaster():
    def __init__(self, server_list=[]):
        self.pub_sock = {}
        for server in server_list:

            print(server)
            self.pub_sock[server] = create_socket_pub(port=server_dict[server][0])
            self.pub_sock[server].setsockopt(zmq.SNDHWM, 20)


    def send(self, server, pm_data, isBytes = True):
        #print(self.pub_sock)
        if not isBytes:
            self.pub_sock[server].send_string(pm_data, zmq.DONTWAIT) 
        else:
            self.pub_sock[server].send(pm_data, zmq.DONTWAIT) 



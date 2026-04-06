import zmq
import threading
import time
import multiprocessing


### server list 
server_dict = {'ch_can':   8201, 
               'pv_can':   8202,
               'send_can': 8203,
               'control':  8204,
               'cam':      8228,
               'cam1':      9228,
               'rear_cam':      8229,
               'calibrate_request': 8301,
               'read_request': 8302,
               'calibrate_response': 8303,
               'log_request': 8304,
               'measurement': 8305,
               'interal_signals': 8306,
               'carla_udp': 8308,
               'controlCommonds': 8309,
}

server_freq = {'ch_can':   50, 
               'pv_can':   50,
               'send_can': 50,
               'control':  50,
               'cam':      20,
               'cam1':      20,
               'rear_cam':      20,
               'calibrate_request': 0,
               'read_request': 50,
               'calibrate_response': 0,
               'log_request': 0,
               'measurement': 0,
               'interal_signals': 0,
               'carla_udp': 50,
               'controlCommonds': 50,
}


def sec_since_boot():
  return time.time()

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

def create_socket_sub(port=5555):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE,''.encode('utf-8'))  # 接收所有消息
    socket.setsockopt(zmq.CONFLATE, True)
    socket.connect("tcp://localhost:" + str(port))   #多个客户端连接同样的地址

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
        self.avgHistoryLen = 100

        self.poller = zmq.Poller()
        self.server_list = server_list
        self.poller_list = poller_list

        #backend threads for non_block server
        for server in server_list:
            self.threads[server] = create_socket_sub(port=server_dict[server])
            self.data[server] = None
            self.recvTime[server] = 0
            self.recvDeltaTm[server] = [0 for i in range(self.avgHistoryLen)]
            self.freq[server] = server_freq[server]
            self.alive[server] = False

        #poller blocking server
        for server in poller_list:
            self.socks[server] = create_socket_sub(port=server_dict[server])
            self.poller.register(self.socks[server], zmq.POLLIN)
            self.data[server] = None

        self.ss = None

    def alive_check(self, check_time=0.2):
        pass

    def stop(self):
        for key, value in self.threads:
            value.terminate()

    def update(self, timeout=1, isDict=True, aliveFactor = 0.8):
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
                if isDict:
                    self.data[s] = eval(data.decode('utf-8'))
                else:
                    self.data[s] = data
                # assign time
                if self.recvTime[s] > 0.001:
                    self.recvDeltaTm[s].append(currentTime - self.recvTime[s])
                    self.recvDeltaTm[s].pop(0)
                self.recvTime[s] = currentTime

            # caculate alive status
            #print(time.time() - currentTime)
            if self.freq[s] > 0.01:
              # alive if delay is within 10x the expected frequency
              self.alive[s] = (currentTime - self.recvTime[s]) < (10. / self.freq[s])

              # alive if average frequency is higher than 90% of expected frequency
              avg_dt = sum(self.recvDeltaTm[s]) / self.avgHistoryLen
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
            self.pub_sock[server] = create_socket_pub(port=server_dict[server])
            self.pub_sock[server].setsockopt(zmq.SNDHWM, 20)


    def send(self, server, pm_data, isBytes = True):
        #print(self.pub_sock)
        if not isBytes:
            self.pub_sock[server].send_string(pm_data, zmq.DONTWAIT) 
        else:
            self.pub_sock[server].send(pm_data, zmq.DONTWAIT) 



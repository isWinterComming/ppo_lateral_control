#!/usr/bin/python3
import time
import pygame
import cv2
import numpy as np
import math
import socket
import json
from typing import Optional, List, Union, Dict

class PyGameDisp():
    def __init__(self, W, H):
        pygame.init()
        pygame.display.set_caption('OpenCV Video Player on Pygame')
        self.screen = pygame.display.set_mode((W, H), 0, 32)
        self.screen.fill([0,0,0])

    def update(self, image, ts=0.01):
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = cv2.transpose(frame)
        frame = pygame.surfarray.make_surface(frame)
        self.screen.blit(frame, (0,0))
        pygame.display.update()
        time.sleep(ts)


class PltgBrige():
    def __init__(self, port=8229):
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.port = port

    def publish(self, dict_data):
        ### send data
        # print(dict_data)
        Message = json.dumps(dict_data)
        bytes_data = Message.encode('utf-8')
        print(len(bytes_data))
        self.udp_sock.sendto(bytes_data, ('127.0.0.1', self.port))

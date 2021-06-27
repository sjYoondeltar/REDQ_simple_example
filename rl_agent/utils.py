
import numpy as np
import random
import os
import math


class Rewardrecorder:
    def __init__(self):
        self.memory = []

    def push(self, data):
        self.memory.append(data)

    def save(self, save_path):
        np.save(os.path.join(save_path, "reward_plot.npy"), self.memory)


import numpy as np
import random
import os
import math


class REWARDRECORDER:
    def __init__(self):
        self.memory = []

    def push(self, r, is_training):
        self.memory.append((r, is_training))

    def save(self, save_path):
          
        np.save(os.path.join(save_path, "reward_plot.npy"), self.memory)

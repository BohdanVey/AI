import pandas as pd
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import random


class DataSampler(Sampler):
    def __init__(self, csv_file, get_from='sampler_weights.txt'):
        if os.path.exists(get_from):
            file = open(get_from, 'r')
            self.arr = [int(x.strip()) for x in file.readlines()]
            file.close()
            random.shuffle(self.arr)
            print(len(self.arr), "!!!!!")
            return
        self.bucket = []
        for i in range(6):
            self.bucket.append([])
        self.csv_path = pd.read_csv(csv_file.params.csv_path)
        self.labels_path = csv_file.params.labels_path
        for item in range(len(self.csv_path)):
            row = self.csv_path.iloc[item]
            pack_path = row['pack']
            if (not item % 1000):
                print(item / len(self.csv_path))
            _, _, _, _, _, sw, cs, ps, wc, ww, dp = np.load(pack_path)

            if np.unique(cs).shape[0] == 2:
                self.bucket[0].append(item)
            if np.unique(dp).shape[0] == 2:
                self.bucket[1].append(item)
            if np.unique(ps).shape[0] == 2:
                self.bucket[2].append(item)
            if np.unique(sw).shape[0] == 2:
                self.bucket[3].append(item)
            if np.unique(ww).shape[0] == 2:
                self.bucket[4].append(item)
            if np.unique(wc).shape[0] == 2:
                self.bucket[5].append(item)
        sampler_weights_file = open(get_from, 'w')
        self.generate_array()
        for i in range(len(self.arr)):
            sampler_weights_file.write(str(self.arr[i]) + '\n')
        sampler_weights_file.close()

    def generate_array(self):
        # [931, 1761, 270, 815, 1769, 8890]
        self.weights = [3, 2, 4, 3, 2, 1]
        self.arr = []
        for i in range(6):
            for x in self.bucket[i]:
                for j in range(self.weights[i]):
                    self.arr.append(x)
        random.shuffle(self.arr)
        self.bucket = []
        print(len(self.arr))

    def __iter__(self):
        return iter(self.arr)

    def __len__(self):
        return len(self.arr)

    def randomize(self):
        random.shuffle(self.arr)

    @classmethod
    def from_config(cls, data):
        return cls(
            csv_file=data
        )


class DataSamplerPlant(DataSampler):
    def __init__(self, csv_file, get_from='plant_train.txt'):
        super().__init__(csv_file, get_from)

    def generate_array(self):
        el = 0
        # [931, 1761, 270, 815, 1769, 8890]
        self.weights = [0, 1, 3, 0, 0, 0]
        self.arr = []
        for i in range(6):
            for x in self.bucket[i]:
                for j in range(self.weights[i]):
                    self.arr.append(x)
        random.shuffle(self.arr)
        self.bucket = []
        print(len(self.arr))


class DataSamplerPlantSmall(DataSampler):
    def __init__(self, csv_file, get_from='plant_validation.txt'):
        super().__init__(csv_file, get_from)

    def generate_array(self):
        el = 0
        self.weights = [0, 1, 1, 0, 0, 0]
        self.arr = []
        for i in range(6):
            for x in self.bucket[i]:
                for j in range(self.weights[i]):
                    self.arr.append(x)
        random.shuffle(self.arr)
        self.bucket = []
        print(len(self.arr))

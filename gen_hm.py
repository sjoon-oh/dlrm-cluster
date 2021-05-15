'''
DLRM Facebookresearch Heat Map
author: sjoon-oh @ Github
source: -
'''

import torch
import torch.nn as nn

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# import copy


#
# generate heatmap
# Returns map
class HeatMapFeature:
    def __init__(self, h_size):
        self.is_init = True
        
        self.h_size = h_size
        # self.label = list(range(h_size))
        self._map = np.zeros((h_size, h_size))
        self._hist = []
        self._sum = np.zeros((h_size, h_size))
    
    def reset(self):
        self._map = np.zeros((self.h_size, self.h_size))

    def set_label(self, label):
        self.label = label

    def record_map(self, batch):
        
        # Batch is 1-D Tensor.
        if self.is_init:
            for idx in range(len(batch)):
                for sidx in range(idx + 1, len(batch)):
                    if batch[idx] != batch[sidx]:
                        self._map[
                            self.label.index(batch[idx])][
                                self.label.index(batch[sidx])] += 1
                        self._map[
                            self.label.index(batch[sidx])][
                                self.label.index(batch[idx])] += 1

        else: pass

        self._hist.append(self._map)
        self.reset()



    def sum_all(self):
        self._sum = sum(self._hist)


    def show_plt(self):

        print(self._sum)

        self.fig, self.ax = plt.subplots()
        im = self.ax.imshow(self._sum, cmap='magma_r')
        
        cbar = self.ax.figure.colorbar(im, ax=self.ax)
        cbar.ax.set_ylabel("", rotation=-90, va="bottom")

        self.ax.set_xticks(np.arange(self.h_size))
        self.ax.set_yticks(np.arange(self.h_size))

        self.ax.set_xticklabels(self.label)
        self.ax.set_yticklabels(self.label)

        self.ax.grid(which="minor",color="w", linestyle='-', linewidth=3)

        plt.setp(self.ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

        self.ax.set_title(f"Heat Map of Single Batch")        
        self.fig.tight_layout()
        plt.show()

        self.is_init = False
        

if (__name__ == '__main__'):
    sample_data = np.random.randint(0, 26, (26, 26)) # Random

    fig = HeatMapFeature(sample_data.shape[0])
    for row in sample_data:
        fig.record_map(row)
    # fig.record_map(sample_data[0])

    fig.sum_all()
    fig.show_plt()



        

    


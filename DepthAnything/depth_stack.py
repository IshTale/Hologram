import numpy as np


class DepthStack:
    def __init__(self, num_layers=5):
        self.num_layers = num_layers

    def generate_layers(self, depth_map):
        layers = []

        for i in range(self.num_layers):
            low = int(i * 256 / self.num_layers)
            high = int((i + 1) * 256 / self.num_layers)

            mask = np.zeros_like(depth_map, dtype=np.uint8)

            if i == self.num_layers - 1:
                mask[(depth_map >= low) & (depth_map <= 255)] = 255
            else:
                mask[(depth_map >= low) & (depth_map < high)] = 255

            layers.append(mask)

        return layers
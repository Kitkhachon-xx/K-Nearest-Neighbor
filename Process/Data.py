import numpy as np
from matplotlib.colors import ListedColormap

class Data:
    def __init__(self, data_x: np.ndarray, data_y: np.ndarray, data_label: np.ndarray,
                 colors: list = ['red', 'blue', 'green']):
        self.x = data_x
        self.y = data_y
        self.label = data_label            # <-- กลุ่ม 0/1/2 แยกจาก y โดยตรง
        self.color_list = colors
        self.cmap = ListedColormap(colors)

    def __len__(self):
        return len(self.x)


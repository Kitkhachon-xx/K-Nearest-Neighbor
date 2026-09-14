import numpy as np
import pandas as pd
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

    def show_data(data_x, data_y, data_label, label_names):
        """
        แสดงข้อมูลดิบ (x, y, label) พร้อมชื่อกลุ่ม
        """
        df = pd.DataFrame(
            {
                "point": range(len(data_x)),
                "x": data_x,
                "y": data_y,
                "label": data_label,
                "name": [label_names[lbl] for lbl in data_label]
            }
        )
        print(df)
    

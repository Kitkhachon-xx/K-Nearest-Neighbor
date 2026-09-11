import numpy as np
class KNNCalculate:

    def __init__(self, data_x: np.ndarray, data_y: np.ndarray, data_label: np.ndarray,
                 label_names: dict = None):
        """
        label_names: dict ที่ map label (ตัวเลข) -> ชื่อกลุ่ม เช่น {0: "Low", 1: "Medium", 2: "High"}
                     ถ้าไม่ใส่ (None) predict_label จะคืนค่าเป็นตัวเลขเหมือนเดิม
        """
        self.data_x = data_x
        self.data_y = data_y
        self.data_label = data_label
        self.label_names = label_names

    def calculate_distance(self, point_x: float, point_y: float) -> np.ndarray:
        """
        Calculate the Euclidean distance from a given point (point_x, point_y)
        to all points in the dataset.
        """
        distances = np.sqrt((self.data_x - point_x) ** 2 + (self.data_y - point_y) ** 2)
        return distances

    def get_k_nearest_neighbors(self, point_x: float, point_y: float, k: int) -> np.ndarray:
        """
        Get the labels of the k nearest neighbors to a given point (point_x, point_y).
        """
        distances = self.calculate_distance(point_x, point_y)
        nearest_indices = np.argsort(distances)[:k] # sorting data by distance form nearest to farthest so it return with index of value for mapping with lables name
        return self.data_label[nearest_indices] # return lables value of nearest sorting indexes

    def _predict_numeric_label(self, point_x: float, point_y: float, k: int):
        """
        เหมือน predict_label แต่คืนค่าเป็นตัวเลข label เสมอ (ไม่ map เป็นชื่อ)
        ใช้ตอนที่ต้องการค่าตัวเลขไปคำนวณ/วาดกราฟ เช่นใน plot_decision_boundary
        """
        neighbors_labels = self.get_k_nearest_neighbors(point_x, point_y, k)
        unique_labels, counts = np.unique(neighbors_labels, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]
        return majority_label

    def predict_label(self, point_x: float, point_y: float, k: int):
        """
        Predict the label of a given point (point_x, point_y) based on the majority
        label of its k nearest neighbors.

        คืนค่าเป็นชื่อกลุ่ม (str) ถ้าตอน __init__ ใส่ label_names ไว้
        ถ้าไม่ได้ใส่ label_names จะคืนค่าเป็นตัวเลข label เหมือนเดิม
        """
        majority_label = self._predict_numeric_label(point_x, point_y, k)

        if self.label_names is not None:
            return self.label_names.get(majority_label, majority_label)
        return majority_label

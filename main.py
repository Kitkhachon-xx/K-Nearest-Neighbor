import numpy as np
from Process.Data import Data
from Process.Calcuate import KNNCalculate
from Process.Plot import Plot

if __name__ == "__main__":
    data_x = np.array([1, 2, 3, 4])
    data_y = np.array([0.5, 1, 0, 2.5])
    data_label = np.array([0, 1, 0, 2])     # กำหนดกลุ่มเอง (หรือมาจาก quantile/KMeans ตามที่คุยกันก่อนหน้า)
    label_names = {0: "Streamer", 1: "Gamer", 2: "E-Sports"}   # แก้ชื่อกลุ่มตรงนี้ให้ตรงกับความหมายจริง

    data = Data(data_x, data_y, data_label)
    for i in range(len(data_x)):
        print(f"Point {i}: x={data_x[i]}, y={data_y[i]}, label={data_label[i]} ({label_names[data_label[i]]})")
    print(len(data))
    Plot.plot_data(data)

    knn_calculator = KNNCalculate(data_x, data_y, data_label, label_names=label_names)
    test_point_x = 2.5
    test_point_y = 2
    k = 3

    Plot.plot_decision_boundary(knn_calculator, k, test_point=(test_point_x, test_point_y))
    
    predicted_label = knn_calculator.predict_label(test_point_x, test_point_y, k)
    print(f"Predicted label for point ({test_point_x}, {test_point_y}) with k={k}: {predicted_label}")
import numpy as np
from Process.Data import Data
from Process.Calculate import KNNCalculate
from Process.Plot import Plot
from Load_utils.load_config import load_config

if __name__ == "__main__":
    config = load_config("config_dev.yaml")
    data_x = np.array(config.get("collect_data", {}).get("x", [1, 2, 3, 4]))
    data_y = np.array(config.get("collect_data", {}).get("y", [0.5, 1, 0, 2.5]))
    data_label = np.array(config.get("collect_data", {}).get("data_labels", [0, 1, 0, 2]))     # กำหนดกลุ่มเอง (หรือมาจาก quantile/KMeans ตามที่คุยกันก่อนหน้า)
    label_names = config.get("collect_data", {}).get("labels_name", {0: "Streamer", 1: "Gamer", 2: "E-Sports"})   # แก้ชื่อกลุ่มตรงนี้ให้ตรงกับความหมายจริง

    data = Data(data_x, data_y, data_label)
    for i in range(len(data_x)):
        print(f"Point {i}: x={data_x[i]}, y={data_y[i]}, label={data_label[i]} ({label_names[data_label[i]]})")
    print(len(data))
    Plot.plot_data(data)

    knn_calculator = KNNCalculate(data_x, data_y, data_label, label_names=label_names)
    test_point_x = config.get("input_data", {}).get("x", 2.5)
    test_point_y = config.get("input_data", {}).get("y", 2)
    k = config.get("input_data", {}).get("k", 3)

    Plot.plot_decision_boundary(knn_calculator, k, test_point=(test_point_x, test_point_y))

    predicted_label = knn_calculator.predict_label(test_point_x, test_point_y, k)
    print(f"Predicted label for point ({test_point_x}, {test_point_y}) with k={k}: {predicted_label}")
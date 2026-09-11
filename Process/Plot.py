import numpy as np
import matplotlib.pyplot as plt


class Plot:
    """
    รวม Function สำหรับวาดกราฟทั้งหมดของโปรเจกต์
    (ย้ายมาจาก Data.plot() และ KNNCalculate.plot_decision_boundary())
    """

    @staticmethod
    def plot_data(data):
        """
        วาดกราฟจุดข้อมูลดิบ (เดิมคือ Data.plot())
        รับ instance ของ Data
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(data.x, data.y, c=data.label, cmap=data.cmap, edgecolor='k')
        plt.xlabel('hours streamed per week')
        plt.ylabel('hours competing per week')
        plt.title('Data Points (3 Groups)')
        plt.colorbar(ticks=sorted(set(data.label)), label='Group')
        plt.show()

    @staticmethod
    def plot_decision_boundary(knn, k: int, x_range: tuple = None, y_range: tuple = None,
                                resolution: int = 100, test_point: tuple = None):
        """
        วาด decision boundary ของ KNN (เดิมคือ KNNCalculate.plot_decision_boundary())
        รับ instance ของ KNNCalculate เป็น knn

        test_point: (x, y) ถ้าใส่มา จะวาดจุด test พร้อมเส้นเชื่อมไปยัง k เพื่อนบ้านที่ใกล้ที่สุด
                    และไฮไลต์เพื่อนบ้านเหล่านั้นบนกราฟ
        """
        if x_range is None:
            x_min, x_max = knn.data_x.min() - 1, knn.data_x.max() + 1
        else:
            x_min, x_max = x_range

        if y_range is None:
            y_min, y_max = knn.data_y.min() - 1, knn.data_y.max() + 1
        else:
            y_min, y_max = y_range

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        # ใช้ _predict_numeric_label (ไม่ใช่ predict_label) เพราะ contourf ต้องการค่าตัวเลข
        # ถ้าใช้ predict_label ตอนมี label_names จะได้ string มา ทำให้ contourf พังแบบ error ที่เจอ
        predictions = np.array([knn._predict_numeric_label(x, y, k) for x, y in grid_points])
        predictions = predictions.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(xx, yy, predictions, alpha=0.3)
        plt.scatter(knn.data_x, knn.data_y, c=knn.data_label, edgecolor='k', cmap=plt.cm.coolwarm)
        plt.xlabel('hours streamed per week')
        plt.ylabel('hours competing per week')
        plt.title(f'KNN Decision Boundary (k={k})')

        if knn.label_names is not None:
            # แปะชื่อกลุ่มบน colorbar แทนตัวเลข label
            unique_labels = sorted(knn.label_names.keys())
            cbar = plt.colorbar(contour, ticks=unique_labels)
            cbar.ax.set_yticklabels([knn.label_names[label] for label in unique_labels])

        if test_point is not None:
            test_x, test_y = test_point
            distances = knn.calculate_distance(test_x, test_y)
            nearest_indices = np.argsort(distances)[:k]

            # เส้นประเชื่อมจากจุด test ไปยัง k เพื่อนบ้านที่ใกล้ที่สุด
            for idx in nearest_indices:
                plt.plot([test_x, knn.data_x[idx]], [test_y, knn.data_y[idx]],
                         color='black', linestyle='--', linewidth=1, zorder=2)

            # ไฮไลต์จุดเพื่อนบ้าน k ตัวที่ใกล้ที่สุด ด้วยวงกลมล้อมรอบ
            plt.scatter(knn.data_x[nearest_indices], knn.data_y[nearest_indices],
                        s=250, facecolors='none', edgecolors='black', linewidths=2,
                        label=f'{k} Nearest Neighbors', zorder=3)

            # จุด test point เอง
            predicted = knn.predict_label(test_x, test_y, k)
            plt.scatter(test_x, test_y, marker='*', s=350, c='yellow', edgecolors='black',
                        linewidths=1.5, label=f'Test point -> {predicted}', zorder=4)

            plt.legend(loc='best')

        plt.show()

""" import math

def distance_between_points(point1, point2):
    # point1 和 point2 分别是两个点的坐标，每个点的坐标以列表形式表示 [x, y, z]
    # 使用三维空间中两点之间的距离公式计算距离
    distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)
    return distance

# 两个点的坐标
point1 = [-168.358, 33.5112, -2277.15]
point2 =  [-200.454, 26.708, -2372.79]

# 调用函数计算两点之间的距离
distance = distance_between_points(point1, point2)

print("两点之间的距离约为:", distance)
 """

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 定义六个点的坐标和名称
points = {
    "AP_center": np.array([-432.725, 58.021, -1679.28]),
    "LAT_center": np.array([-200.454, 26.708, -2372.79]),
    "AP_source": np.array([519.904, 77.2454, -2017.68]),
    "LAT_source": np.array([120.501, 94.7396, -1416.4]),
    "AP_rotcenter": np.array([-337.462, 59.9435, -1713.12]),
    "Lat_rotcenter": np.array([-168.358, 33.5112, -2277.15]),
    "FixCenter": np.array([0, 60, -1800.15])

}

# 创建 3D 图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制六个点
for name, point in points.items():
    ax.scatter(point[0], point[1], point[2], label=name, marker='o')

# 连线
ax.plot([points["AP_center"][0], points["AP_source"][0]], [points["AP_center"][1], points["AP_source"][1]], [points["AP_center"][2], points["AP_source"][2]], c='b')
ax.plot([points["LAT_center"][0], points["LAT_source"][0]], [points["LAT_center"][1], points["LAT_source"][1]], [points["LAT_center"][2], points["LAT_source"][2]], c='b')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 添加图例
ax.legend()

# 显示图形
plt.show()

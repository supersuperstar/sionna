import numpy as np
def is_point_inside_polygon(polygon, point):
    """
    判断点是否在多边形内部
    :param polygon: 多边形顶点列表，每个顶点是一个(x, y)元组
    :param point: 需要判断的点，格式为(x, y)
    :return: 如果点在多边形内部返回True，否则返回False
    """
    count = 0
    x, y = point
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if y1 == y2:  # 水平边，忽略
            continue

        if y < min(y1, y2):  # 点在边下方，忽略
            continue

        if y >= max(y1, y2):  # 点在边上方，忽略
            continue

        # 计算交点的x坐标
        x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1

        if x_intersect > x:  # 交点在点的右侧
            count += 1

    return count % 2 == 1

# 定义多边形顶点列表
polygon_points = [(1, 2), (1, 1), (2, 1), (2, -1), (1, -1), (1, -2), (0, -2), (0, 2)]

# 测试点
test_point = (1,1)  # 可以更改这个点来进行测试

# 调用函数判断
is_inside = is_point_inside_polygon(polygon_points, test_point)
print(is_inside)

print(np.where([0,1,1,0,0,1]))



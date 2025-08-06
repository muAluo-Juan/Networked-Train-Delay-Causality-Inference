import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

'''
获得k阶邻居
'''
def get_k_hop_neighbors(G, node, k):  # 获得k跳节点集合
    if k == 0:
        return {node}
    else:
        return (set(nx.single_source_dijkstra_path_length(G, node, k).keys())
                - set(nx.single_source_dijkstra_path_length(
                    G, node, k - 1).keys()))




'''
将超出24小时的时间字符串进行重新处理，得加到天数上
'''
def convert_datetime_with_hours(timestamp):
    # 拆分日期和时间部分
    date_part, time_part = timestamp.split(' ')
    hours, minutes, seconds = map(int, time_part.split(':')) # 生成数据集使用

    # hours, minutes = map(int, time_part.split(':'))  # 处理实际数据使用
    seconds = 0

    # 计算累加后的天数和小时
    extra_days = hours // 24
    remaining_hours = hours % 24

    # 将日期部分转换为 datetime 对象，并累加天数
    datetime_obj = pd.to_datetime(date_part) + pd.Timedelta(days=extra_days)

    # 构造新的时间戳，并返回
    new_timestamp = datetime_obj + pd.Timedelta(hours=remaining_hours,
                                                minutes=minutes,
                                                seconds=seconds)
    return new_timestamp


def history_of(A_norm, t, s, k, Th, X):
    events = []
    for index in range(len(X)):  # X中装的是比当前事件先发生的事件
        adj_s = X[index][1]
        adj_t = X[index][3]
        aa = A_norm[k][s, adj_s]
        bb = t - adj_t
        if A_norm[k][s, adj_s] > 0 and adj_t < t and t - adj_t <= Th:
            events.append(index)
            print("和历史事件的时间差：", t-adj_t)
    return events


'''
统计两个节点间的路径数量
'''
def count_paths(A, max_length):
    n = len(A)
    total_paths = np.zeros((n, n), dtype=int)
    for k in range(1, max_length + 1):
        A_k = matrix_power(A, k)
        total_paths += A_k
    return total_paths


'''
计算k阶矩阵
'''
def matrix_power(A, k):
    return np.linalg.matrix_power(A, k)
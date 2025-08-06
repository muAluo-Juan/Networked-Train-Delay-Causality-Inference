import json
import os
import random
import numpy as np
import pandas as pd
from model.MTHPSimulator import MTHPSimulation
from utils.util import convert_datetime_with_hours

'''
设置生成的数据存储路径
'''
# generate_path_prefix = 'synthetic_data/2019-11-08_01-07/mthp_jh_jg_xl_hk_150/'
generate_path_prefix = '../synthetic_data/2019-10-08_11-07/mthp_jh_4/'

'''
读取整个铁路网络的运营数据（数据有误，同一天内有同一个列车经过同一个站点的数据）
'''
# train_operation = pd.read_csv("raw_data/high-speed trains operation data.csv")
# 京沪线
train_operation = pd.read_csv("../raw_data/j-h_line.csv")
# train_operation = pd.read_csv("../raw_data/jh_jg_xl_hk_lines.csv")

'''
提取某一天的列车运营数据用于生成到站晚点事件数据集
'''
# train_operation = train_operation.loc[(train_operation['date'] >= '2019-10-08')
#                                       & (train_operation['date'] <= '2019-11-07')
#                                       & (train_operation['scheduled_arrival_time'] >= "06:00:00")
#                                       & (train_operation['scheduled_arrival_time'] <= "20:00:00")]

# train_operation = train_operation.loc[(train_operation['date'] >= '2019-11-08')
#                                       & (train_operation['date'] <= '2020-01-07')]

train_operation = train_operation.loc[(train_operation['date'] >= '2019-10-08')
                                      & (train_operation['date'] <= '2019-11-07')]


'''
提取到站时刻表及到站事件
'''
arrival_events = train_operation[['date',
                                  'train_number',
                                  'train_direction',
                                  'station_name',
                                  'station_order',
                                  'scheduled_arrival_time',
                                  'actual_arrival_time',
                                  'stop_time',
                                  'arrival_delay',
                                  'wind',
                                  'weather']]
arrival_events['scheduled_datetime'] = arrival_events['date'] + ' ' + arrival_events['scheduled_arrival_time']
arrival_events['scheduled_datetime'] = arrival_events['scheduled_datetime'].apply(convert_datetime_with_hours)
arrival_events['scheduled_datetime'] = pd.to_datetime(arrival_events['scheduled_datetime']).astype('int64') / (
        10 ** 9 * 60)  # 分钟为单位，图定到站/发车时间
arrival_events['actual_datetime'] = arrival_events['date'] + ' ' + arrival_events['actual_arrival_time']
arrival_events['actual_datetime'] = arrival_events['actual_datetime'].apply(convert_datetime_with_hours)
arrival_events['actual_datetime'] = pd.to_datetime(arrival_events['actual_datetime']).astype('int64') / (
        10 ** 9 * 60)  # 分钟为单位，图定到站/发车时间
arrival_events = arrival_events.sort_values(by='scheduled_datetime')  # 排序，正常按时刻表执行
arrival_events.to_csv(generate_path_prefix + "arrival_events.csv")


'''
提取前150列车的数据
'''
unique_trains = arrival_events.drop_duplicates('train_number')['train_number'].values[0: 150]
arrival_events = arrival_events[arrival_events['train_number'].isin(unique_trains)]


'''
一些数据设置
'''
K = 2  # 相邻站点阶数
Dates = arrival_events['date'].values  # 生成晚点事件的日期

'''
获得V
'''
trains = arrival_events.drop_duplicates('train_number')['train_number'].values
keys = trains
values = [i for i in range(len(trains))]
train_pairs = {key: value for key, value in zip(keys, values)}  # 列车和编号对应
with open(generate_path_prefix + 'trains.json', 'w') as file:
    json.dump(train_pairs, file)

'''
获得S
'''
stations = arrival_events.drop_duplicates('station_name')["station_name"].values
keys = stations
values = [i for i in range(len(stations))]  # 所有运营站点，站点从编号0开始
station_pairs = {key: value for key, value in zip(keys, values)}  # 站点和编号对应
with open(generate_path_prefix + 'stations.json', 'w') as file:
    json.dump(station_pairs, file)

'''
获得N_S、N_V
'''
N_S = len(stations)
N_V = len(trains)
Th = 240  # 考虑多远的历史事件的影响

'''
获取站点对应顺序的列车、列车对应顺序的站点、列车对应的图定到站时间
'''
train_schedule_times = dict()  # 每天每个列车到达各站点的图定时间
train_stations = dict()   # 每天每个列车按时刻表到达的各个站点
temp_station_trains = dict()  # 每天每个站点按时刻表到达的各个列车
for s in range(N_S):
    temp_station_trains[s] = []
for i, day in enumerate(Dates):
    train_schedule_times[day] = dict()
    train_stations[day] = dict()
    for v in range(N_V):
        train_schedule_times[day][v] = []
        train_stations[day][v] = []
for index, i in enumerate(np.array(arrival_events)):  # 站点、列车、事件编号均从0开始
    day, train_id, station_id, station_order = i[0], train_pairs[i[1]], station_pairs[i[3]], i[4]
    schedule_t = i[11]  # 新增的一列
    train_schedule_times[day][train_id].append(schedule_t)
    train_stations[day][train_id].append(station_id)
    temp_station_trains[station_id].append(train_id)

'''
提取高铁站点拓扑网络作为G_S(无向有环图)
'''
adjacent_stations = pd.read_csv("../raw_data/adjacent railway stations mileage data.csv")
if not os.path.exists(generate_path_prefix + 'G_S_temp.npy'):
    G_S = np.eye(len(stations), dtype=int)  # 保证自环
    for i in np.array(adjacent_stations):
        if i[0] not in station_pairs or i[1] not in station_pairs:
            continue
        from_s = station_pairs[i[0]]
        to_s = station_pairs[i[1]]
        if i[0] in keys and i[1] in keys:
            G_S[to_s][from_s] = 1
            G_S[from_s][to_s] = 1
    np.save(generate_path_prefix + 'G_S.npy', G_S)
else:
    G_S = np.load(generate_path_prefix + 'G_S.npy')

'''
得到0到K阶邻接矩
阵（无向有环图）
'''
if not os.path.exists(generate_path_prefix + 'A.npy'):
    A = []  # 0~K阶01邻接矩阵
    for k in range(K + 1):
        if k == 0:  # 0阶,单位矩阵,每个站点和自身是连通的(每个站点内的列车和事件可以相互影响)
            adj = np.eye(N_S)
        elif k == 1:
            adj = G_S
        else:
            adj = (np.linalg.matrix_power(G_S, k) > 0).astype(int)  # 转为0-1矩阵了
        A.append(adj)
    np.save(generate_path_prefix + 'A.npy', A)
else:
    A = np.load(generate_path_prefix + 'A.npy')

'''
生成随机有向有环图G_V
'''
if not os.path.exists(generate_path_prefix + 'G_V_temp.npy'):
# 得到列车V邻接矩阵（在K阶相邻站点的认为相邻，不相邻站点的则认为不相邻，重点看不相邻的列车，这不是最终使用的因果图）
    G_V = np.eye((len(trains)), dtype=int)  # 列车自身和自身一定存在关联
    for k in range(K + 1):
        s_adj = A[k]
        for i in range(len(s_adj)):
            for j in range(len(s_adj[i])):
                if s_adj[i][j] == 1:  # 如果两个站点相邻，可以认为在这两个站点的两个列车也相邻
                    # 找到i站点上的列车
                    i_trains = temp_station_trains[i]
                    # 找到j站点上的列车
                    j_trains = temp_station_trains[j]
                    # 遍历i站和j站的列车，随机使其邻接矩阵为1，b被a影响表示为(b,a)
                    for a in i_trains:
                        for b in j_trains:
                                G_V[b][a] = G_V[a][b] = 1
    indices = np.argwhere((G_V == 1) & (np.arange(G_V.shape[0])[:, None] != np.arange(G_V.shape[1])))
    # 随机选择需要置为0的位置
    num_to_zero = int(len(indices) * 0.7)
    random_indices = random.sample(range(len(indices)), num_to_zero)
    for index in random_indices:
        i, j = indices[index]
        G_V[i, j] = 0
    np.save(generate_path_prefix + 'G_V_temp.npy', G_V)
else:
    G_V = np.load(generate_path_prefix + 'G_V_temp.npy')

'''
构建数据生成器
'''
simulator = MTHPSimulation(G_V, G_S,
                          # mu_range=(0.0008, 0.01),
                          mu_range=(0.001, 0.01),
                          alpha_range=(0.5, 1),
                          beta_range=(0.005, 0.05))
X = simulator.simulate(K, N_V, train_stations, train_schedule_times,
                       A, generate_path_prefix, Th)

print(X)
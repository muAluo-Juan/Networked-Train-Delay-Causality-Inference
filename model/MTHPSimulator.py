import pickle
import networkx as nx
import numpy as np
import pandas as pd
from utils.util import history_of
from datetime import datetime


class MTHPSimulation(object):
    def __init__(self, causal_matrix, topology_matrix,
                 mu_range, alpha_range, beta_range):
        assert (isinstance(causal_matrix, np.ndarray) and
                causal_matrix.ndim == 2 and
                causal_matrix.shape[0] == causal_matrix.shape[1]), \
            'casual_matrix should be np.matrix object, two dimension, square.'
        assert (isinstance(topology_matrix, np.ndarray) and
                topology_matrix.ndim == 2 and
                topology_matrix.shape[0] == topology_matrix.shape[1]), \
            'topology_matrix should be np.matrix object, two dimension, square.'

        self._causal_matrix = (causal_matrix != 0).astype(int)  # 转为0-1矩阵

        self._topo = nx.Graph(topology_matrix)

        self._mu_range = mu_range
        self._alpha_range = alpha_range
        self._beta_range = beta_range

    def simulate(self, K, N_V, train_stations, train_schedule_times,
                 A, generate_path_prefix, Th):
        """
        Generate simulation data.
        """

        mu = np.random.uniform(*self._mu_range, N_V)

        beta = np.random.uniform(*self._beta_range, N_V)

        alpha = np.random.uniform(*self._alpha_range, [N_V, N_V])
        alpha = alpha * self._causal_matrix
        alpha = np.ones([K + 1, N_V, N_V]) * alpha

        X = []  # 存储晚点事件
        temp_id = 0
        ini_events = []  # 初始晚点事件id
        tri_events = []  # 连带晚点事件id
        g_x = dict()  # 存储影响的事件,key影响value，所有天数的包含在一起了
        event_orders = dict()  # 记录每个列车发生了晚点的站点(一天内该列车在该站发生晚点,则不会再发生晚点)
        for day in train_stations.keys():
            event_orders[day] = dict()
            for v in range(N_V):
                event_orders[day][v] = []

        big = 0
        small = 0
        ini_big = 0
        ini_small = 0
        error = 0
        N_ini_v = 2  # 规定每个列车最多生成2个初始晚点事件
        DAYS = train_stations.keys()
        for day in DAYS:
            '''
            生成初始晚点事件
            '''
            for v in range(N_V):
                v_stations = train_stations[day][v]
                v_sches = train_schedule_times[day][v]
                if len(v_stations) < 1:  # 列车并不是在每一天都出现
                    continue
                trigger_time = v_sches[0]

                '''
                更换一种生成晚点事件的方式
                '''
                event_orders[day][v].append(1)  # 始发站无到站晚点
                mu_v = mu[v]
                n_ini = 0
                while 1:
                    delta_t = round(np.random.exponential(1 / mu_v))
                    temp_time = trigger_time + delta_t
                    if n_ini >= N_ini_v or temp_time > v_sches[-1] + 240 or len(v_stations) in event_orders[day][v]:
                        break

                    for idx, s in enumerate(v_stations):
                        delay = temp_time - v_sches[idx]
                        s_order = idx + 1

                        if s_order in event_orders[day][v] or temp_time <= v_sches[1]:  # 一天一个列车在一个站点最多发生一次晚点事件
                            continue

                        if idx == len(v_stations) - 1:  # 如果发生在最后一个站点，满足晚点时长不超过240的约束
                            schedule_time = v_sches[idx]
                            if temp_time > schedule_time and temp_time - schedule_time <= 240:
                                event_orders[day][v].append(s_order)
                                trigger_time = temp_time
                                print("生成初始晚点事件: ", [v, s, s_order, trigger_time, v_sches[idx], delay, mu[v], temp_id])
                                date_obj = datetime.fromtimestamp(trigger_time * 60)
                                new_day = date_obj.strftime('%Y-%m-%d')
                                X.append([v, s, s_order, trigger_time, v_sches[idx], delay, mu[v], 'ini_delay', new_day,
                                             temp_id])
                                ini_events.append(temp_id)
                                # 更新
                                temp_id += 1
                                n_ini += 1
                                break
                        else:  # 如果不发生在最后一个站点，则需要满足小于下一个时刻表的约束
                            schedule_time = v_sches[idx]
                            next_schedule_time = v_sches[idx + 1]
                            if temp_time > schedule_time and temp_time <= next_schedule_time:
                                event_orders[day][v].append(s_order)
                                trigger_time = temp_time
                                print("生成初始晚点事件: ",
                                      [v, s, s_order, trigger_time, v_sches[idx], delay, mu[v], temp_id])
                                date_obj = datetime.fromtimestamp(trigger_time * 60)
                                new_day = date_obj.strftime('%Y-%m-%d')
                                X.append([v, s, s_order, trigger_time, v_sches[idx], delay, mu[v], 'ini_delay', new_day,
                                          temp_id])
                                ini_events.append(temp_id)
                                # 更新
                                temp_id += 1
                                n_ini += 1
                                break

                # for idx, s in enumerate(v_stations):  # 列车v途径的所有站点s
                #     s_order = idx + 1
                #     if s_order == 1:  # 始发站不发生到站晚点
                #         continue
                #
                #     mu_v = mu[v]
                #     delta_t = round(np.random.exponential(1 / mu_v))
                #     temp_time = trigger_time + delta_t
                #     delay = temp_time - v_sches[idx]
                #
                #     if idx < len(v_stations) - 1:
                #         next_delta = v_sches[idx + 1] - temp_time
                #         if temp_time > v_sches[idx + 1]:  # 假定不超过下一个该列车到站事件的预计到站时间
                #             print("temp_time过大: ",
                #                   [v, s, s_order, temp_time, v_sches[idx], delay, mu[v], temp_id])
                #             ini_big += 1
                #             continue
                #
                #     if delay > 0:
                #         trigger_time = temp_time
                #         print("生成初始晚点事件: ",
                #               [v, s, s_order, trigger_time, v_sches[idx], delay, mu[v], temp_id])
                #         X.append(
                #             [v, s, s_order, trigger_time, v_sches[idx], delay, mu[v], 'ini_delay', day,
                #              temp_id])
                #         ini_events.append(temp_id)
                #         event_orders[day][v].append(s_order)
                #
                #         # 更新
                #         temp_id += 1
                #     else:
                #         print("不满足生成初始晚点事件要求: ",
                #               [v, s, s_order, temp_time, v_sches[idx], delay, mu[v], temp_id])
                #         ini_small += 1


            '''
            生成连带晚点事件
            '''
            for v in range(N_V):
                v_stations = train_stations[day][v]
                v_sches = train_schedule_times[day][v]
                if len(v_stations) < 1:  # 列车并不是在每一天都出现
                    continue
                # trigger_time = v_sches[0]
                for idx, s in enumerate(v_stations):  # 列车v途径的所有站点s
                    s_order = idx + 1
                    if s_order == 1 or s_order in event_orders[day][v]:  # 始发站不发生到站晚点 | 一天一个列车在一个站点最多发生一次晚点事件
                        continue

                    temp_X = pd.DataFrame(X)
                    temp_X_v = temp_X[temp_X[0] == v]
                    temp_X_v = temp_X_v[temp_X_v[8] == day]

                    if not temp_X_v.empty:  # 如果该列车已经有生成的事件
                        temp_t = temp_X_v[3].values  # 筛选同一天的

                        if min(temp_t) > v_sches[idx]:  # 如果已生成的最下的事件时间都比将要生成的事件的schedule大，则没有该列车历史，将当天第一个到站图定时间作为起始点
                            trigger_time = v_sches[idx]
                        else:  # 否则有历史，将本列车的最近一个历史作为起始时间点
                            max_tt = min(temp_t)
                            for id, tt in enumerate(temp_t):
                                if tt >= max_tt and tt <= v_sches[idx]:
                                    max_tt = tt
                            trigger_time = max_tt
                    else:  # 如果不存在已经生成该列车的事件，则将当天该列车的第一个到站图定时间作为起始点
                        trigger_time = v_sches[idx]

                    mu_mu_v = mu[v]
                    lambda_v = mu_mu_v
                    g_x[temp_id] = []
                    k_histories = []
                    for k in range(K + 1):  # 由历史事件触发与基础强度叠加生成的晚点事件
                        # 寻找s的k阶邻居上发生的事件
                        histories = history_of(A, trigger_time, s, k, Th, X)  # 返回X的下标
                        k_histories.append(histories)
                        for j in histories:
                            history_event_v = X[j][0]
                            history_temp_id = X[j][-1]
                            history_timestamp = X[j][3]
                            alpha_k_v_v = alpha[k, v, history_event_v]
                            sum_his_t = (beta[v] * np.exp(- beta[v] * (trigger_time - history_timestamp)))
                            aa = - beta[v]
                            bb = np.exp(- beta[v] * (trigger_time - history_timestamp))
                            # sum_his_t = (np.exp(- beta[v] * (trigger_time - history_timestamp)))
                            x = alpha_k_v_v * sum_his_t

                            if history_event_v == v and alpha_k_v_v == 0:
                                print("cool")

                            lambda_v += x
                            if x > 0:  # 说明两个事件且两个列车确实存在影响
                                g_x[temp_id].append(history_temp_id)

                    delta_t = round(np.random.exponential(1 / lambda_v))
                    temp_time = trigger_time + delta_t
                    delay = temp_time - v_sches[idx]

                    if idx < len(v_stations) - 1:  # 如果不是终点站，约束生成事件的时间不超过该列车下一个到站图定时间
                        if temp_time > v_sches[idx + 1] or delay >= 240:
                            print("temp_time过大: ",
                                  [v, s, s_order, temp_time, v_sches[idx], delay, lambda_v, temp_id])
                            g_x[temp_id] = []
                            big += 1
                            continue

                    if delay > 0:
                        if mu_mu_v == lambda_v:
                            print("这不应该是初始晚点事件吗？",
                                  [v, s, s_order, v_sches[idx], delay, mu[v], temp_id])
                            g_x[temp_id] = []
                            error += 1
                            continue
                        qq = lambda_v - mu_mu_v
                        bi = beta[v]
                        dd = trigger_time - history_timestamp
                        trigger_time = temp_time
                        print("生成连带晚点事件: ",
                              [v, s, s_order, trigger_time, v_sches[idx], delay, lambda_v, temp_id])
                        date_obj = datetime.fromtimestamp(trigger_time * 60)
                        new_day = date_obj.strftime('%Y-%m-%d')
                        X.append(
                            [v, s, s_order, trigger_time, v_sches[idx], delay, lambda_v, 'tri_delay', new_day,
                             temp_id])
                        tri_events.append(temp_id)
                        event_orders[day][v].append(s_order)

                        # 更新
                        temp_id += 1

                        # '''
                        # 待删除，看看是否有邻居事件时间大于或等于本事件时间的情况
                        # '''
                        # for k in range(K + 1):
                        #     for index in k_histories[k]:
                        #         his_t = X[index][3]
                        #         if his_t >= trigger_time:
                        #             raise ValueError(
                        #                 f"history time should smaller than trigger time: {his_t}, {trigger_time}")

                    else:
                        print("不满足生成连带晚点事件要求: ",
                              [v, s, s_order, v_sches[idx], delay, mu[v], temp_id])
                        g_x[temp_id] = []
                        small += 1

        Xn = pd.DataFrame(X, columns=[
            'train_id',
            'station_id',
            'station_order',
            'event_time',
            'schedule_time',
            'delay',
            'lambda',
            'delay_type',
            'date',
            'temp_id'
        ])

        Xn = Xn.sort_values(by='event_time')  # 排序，按事件发生正常时间排序
        Xn.reset_index(drop=True, inplace=True)
        Xn.insert(0, 'event_id', Xn.index)

        '''
        提取temp_id和event_id对应
        '''
        temp_ids = Xn['temp_id'].values
        event_ids = Xn['event_id'].values
        id_pairs = dict(zip(temp_ids, event_ids))

        print("初始晚点的event_id为：", [id_pairs[i] for i in ini_events])
        print("连带晚点的event_id为：", [id_pairs[i] for i in tri_events])

        '''
        提取新的train_id和station_id（生成的晚点事件的列车和站点不一定和真实数据集一致）
        '''
        # new_trains = np.unique(Xn['train'].values)  # temp_train_id
        # new_trains_id = [i for i in range(len(new_trains))]
        # new_trains_id_pairs = {key: value for key, value in
        #                        zip(new_trains, new_trains_id)}  # (temp_train_id, new_train_id)
        # Xn['train'] = Xn['train'].map(new_trains_id_pairs)  # 将事件文件中的列车id更新为新的
        # # np.save(generate_path_prefix + 'new_trains_id_pairs.npy', new_trains_id_pairs)
        #
        # new_stations = np.unique(Xn['station'].values) # temp_station_id
        # new_stations_id = [i for i in range(len(new_stations))]
        # new_stations_id_pairs = {key: value for key, value in
        #                          zip(new_stations, new_stations_id)}  # (temp_station_id, new_station_id)
        # Xn['station'] = Xn['station'].map(new_stations_id_pairs)  # 将事件文件中的站点id更新为新的
        # np.save(generate_path_prefix + 'new_stations_id_pairs.npy', new_stations_id_pairs)

        '''
        输出X
        '''
        Xn = Xn.drop(Xn.columns[-1], axis=1)
        Xn.to_csv(generate_path_prefix + 'events.csv', index=False)  # 获得新的index

        '''
        输出事件因果图(一天的)及列车因果图
        '''
        N_X = len(X)
        G_X = np.zeros((N_X, N_X))
        # new_N_V = len(new_trains)
        # new_G_V = np.eye(new_N_V)
        new_G_V = np.zeros((N_V, N_V))
        vs = Xn['train'].values
        for i in g_x.keys():  # key表示的是被影响的事件
            if len(g_x[i]) > 0:
                to_x = id_pairs[i]
                to_v = vs[to_x]
                for j in g_x[i]:  # values表示的是原因事件
                    from_x = id_pairs[j]
                    from_v = vs[from_x]
                    G_X[to_x][from_x] = 1
                    new_G_V[to_v][from_v] = 1
                    if from_x > to_x:
                        raise ValueError(f'from id should be smaller than to id: {from_x}, {to_x}')
        np.save(generate_path_prefix + 'G_X.npy', G_X)
        np.save(generate_path_prefix + 'G_V.npy', new_G_V)

        '''
        输出站点邻接矩阵
        '''
        # new_A = []
        # new_N_S = len(new_stations)
        # for k in range(K + 1):
        #     if k == 0:  # 0阶,单位矩阵,每个站点和自身是连通的(每个站点内的列车和事件可以相互影响)
        #         adj = np.eye(new_N_S)
        #     else:
        #         adj = np.zeros((new_N_S, new_N_S))
        #         indices = np.where(A[k] > 0)
        #         for i, j in zip(indices[0], indices[1]):  # A中还是旧的站点id
        #             if i in new_stations_id_pairs.keys() and j in new_stations_id_pairs.keys():
        #                 adj[new_stations_id_pairs[i], new_stations_id_pairs[j]] = A[k][i, j]
        #     new_A.append(adj)
        # np.save(generate_path_prefix + 'A.npy', new_A)

        '''
        保存各项参数
        '''
        # 保存alpha
        params = {
            'mu': mu,
            'alpha': alpha,
            'beta': beta,
            'mu_range': self._mu_range,
            'alpha_range': self._alpha_range,
            'beta_range': self._beta_range
        }

        with open(generate_path_prefix + 'params.pk', 'wb') as f:
            pickle.dump(params, f)

        return X
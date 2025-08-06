import json
import math
import os
import numpy as np
from numba import cuda


class TDCausal(object):
    def __init__(self, nu_mu, nu_beta, nu_alpha, eta, omega, Th):
        self.nu_mu = nu_mu
        self.nu_beta = nu_beta
        self.nu_alpha = nu_alpha
        self.eta = eta
        self.omega = omega
        self.Th = Th
        pass

    '''
    验证事件是否按照时间顺序排列
    '''

    def verify_timeorder(self,
                         timestamps,
                         trains,
                         stations,
                         Th):

        if len(trains) != len(timestamps) or len(stations) != len(timestamps):
            raise ValueError('Length differs:timestamp and events')
        elif Th > 1440:
            raise ValueError('Th is too large (cannot be greater than one day)')
        elif Th < 0:
            raise ValueError('Th cannot be negative)')
        elif not isinstance(timestamps, np.ndarray):
            raise ValueError('timestamps must be numpy 1D array')
        elif not isinstance(trains, np.ndarray):
            raise ValueError('trains must be numpy 1D array')
        elif not isinstance(stations, np.ndarray):
            raise ValueError('stations must be numpy 1D array')

        dt_timestamps = (timestamps[1:-1] - timestamps[0:-2])
        isTimeOrdered = ((dt_timestamps < 0).sum() == 0)

        if ~isTimeOrdered:
            print('timestamps must be time-ordered.')
            idxes = timestamps.argsort()
            timestamps = timestamps.copy()[idxes]
            trains = trains.copy()[idxes]
            stations = stations.copy()[idxes]
            print('timestamps are time-ordered.')

        if not isinstance(timestamps[0], float):
            timestamps = np.array(timestamps, dtype='float64')
        if not (timestamps == sorted(timestamps)).all():
            idxes = timestamps.argsort()
            timestamps = timestamps.copy()[idxes]
            trains = trains.copy()[idxes]
            stations = stations.copy()[idxes]

        return timestamps, trains, stations

    '''
    计算自激强度
    '''

    def trigger(self, beta_v, delta_t, alpha_k_v_v_i, A_k_s_s_i):
        xx = alpha_k_v_v_i * A_k_s_s_i * beta_v * np.exp(- beta_v * delta_t)
        yy = np.exp(- beta_v * delta_t)
        return alpha_k_v_v_i * A_k_s_s_i * beta_v * np.exp(- beta_v * delta_t)

    '''
    初始化q
    '''

    def initialize_q(self,
                     mu,
                     timestamps,
                     stations,
                     trains,
                     Delta,
                     histories,
                     histories_hop,
                     beta,
                     alpha,
                     A):

        N = len(timestamps)
        # qself = np.ones(N)  # self-triggering prob.
        qself = np.repeat(np.nan, N)
        qhist = np.repeat(np.nan, N * N).reshape([N, N])

        for n in range(N):  # n=0,...,N-1表达当前事件下标

            mu_n = mu[trains[n]]
            qself[n] = mu_n

            '''
            加快运算的方式
            '''
            history_of_n = histories[n]
            alpha_k_v_v_i = alpha[histories_hop[n], trains[n], trains[history_of_n]]
            A_k_s_s_i = A[histories_hop[n], stations[n], stations[history_of_n]]
            bb = self.trigger(beta[trains[n]], Delta[n], alpha_k_v_v_i, A_k_s_s_i)

            qhist[n, history_of_n] = bb

            rowsum = qself[n] + qhist[n, ~np.isnan(qhist[n, :])].sum()
            qself[n] = qself[n] / rowsum
            qhist[n, :] = qhist[n, :] / rowsum

            if np.isinf(qhist[n, history_of_n]).any() or np.isnan(qhist[n, history_of_n]).any():
                raise ValueError("qself is out of range: ", qhist[n, history_of_n])

        return qself, qhist

    '''
    更新zeta
    '''

    def update_mu(self,
                  timestamps,
                  qself,
                  nu_mu,
                  mu,
                  train_events,
                  days,
                  day_list,
                  day_events,
                  histories,
                  ddtt):

        for v in range(len(mu)):
            v_events = train_events[v]  # 列车v的所有事件

            B_mu = 0

            for n in v_events:  # n是事件的编号
                # if len(histories[n]) == 0:  # 下标为1表示第一个事件，第一个晚点事件没有上一个事件
                #     continue
                if n == 0:
                    continue
                # day = days[n]
                # day_id = day_list.tolist().index(day)
                # 定位该事件在day_events中的位置
                # in_day_idx = day_events[day_id].tolist().index(n)
                # if in_day_idx == 0:
                #     continue
                # last_event_idx = day_events[day_id][in_day_idx - 1]
                # t_ = timestamps[last_event_idx]  # 防止计算到跨天的时间戳
                # delta_t = timestamps[n] - t_
                # if delta_t < 0:
                #     raise ValueError("delta_t is negative")

                # t_ = max(timestamps[histories[n]])
                # delta_t_ = timestamps[n] - t_

                delta_t_ = ddtt[n]

                B_mu += delta_t_

            C_mu = qself[v_events].sum()
            mu[v] = (-B_mu + np.sqrt(B_mu ** 2 + 4. * nu_mu * C_mu)) / (2. * nu_mu)

            # print(mu[v])

            # 让mu_v减小，应该增加nu_mu
            # if math.isinf(mu[v]) or math.isnan(mu[v]) or mu[v] < 0 or mu[v] > 1:
            #     raise ValueError("mu is out of range: ", mu[v])
        return mu

    '''
    时间衰减函数的积分
    '''

    def temporal_decay_integral(self, delta_t, beta_v):
        if delta_t.any() < 0:
            raise ValueError("delta_t is negative")
        return - np.exp(- beta_v * delta_t)

    '''
    时间衰减函数对beta求偏导
    '''

    def beta_derivative(self, delta_t, beta_v):
        if delta_t.any() < 0:
            raise ValueError("delta_t is negative")
        return delta_t * np.exp(- beta_v * delta_t)


    '''
    更新alpha
    '''
    def update_alpha(self,
                     alpha,
                     train_events,
                     stations,
                     trains,
                     timestamps,
                     histories,
                     histories_hop,
                     beta,
                     A,
                     Delta,
                     qhist,
                     H,
                     Q,
                     delta,
                     nu_alpha,
                     omega,
                     days,
                     day_list,
                     day_events,
                     ddtt,
                     based_on_graph,
                     K):

        temp1 = np.zeros((alpha.shape[0], alpha.shape[1], alpha.shape[2]))
        temp2 = np.zeros((alpha.shape[0], alpha.shape[1], alpha.shape[2]))

        for v in range(alpha.shape[1]):
            idx_of_v = train_events[v]  # 列车v的所有运营事件的id
            for n in idx_of_v:
                if n == 0:
                    continue

                without_k_his = histories[n]
                without_k_hop = histories_hop[n]
                this_k = np.unique(without_k_hop)  # 历史邻居事件所在的拓扑阶层

                for k in this_k:
                    idxs = [index for index, value in enumerate(without_k_hop) if value == k]
                    with_k_his = [without_k_his[ii] for ii in idxs]

                    A_k_s_s_i = A[k, stations[n], stations[with_k_his]]

                    Ddelta = np.array([Delta[n][ii] for ii in idxs])

                    Delta_last = Ddelta - ddtt[n]

                    h_n_i = A_k_s_s_i * (self.temporal_decay_integral(Ddelta, beta[v])
                                         - self.temporal_decay_integral(Delta_last, beta[v]))

                    his_trains = np.unique(trains[with_k_his])
                    for vv in his_trains:
                        mask = np.isin(with_k_his, train_events[vv]).astype(bool)  # 定位k阶历史事件中是vv列车的事件
                        temp1[k, v, vv] += h_n_i[mask].sum()
                        temp2[k, v, vv] += qhist[n, with_k_his][mask].sum()

                        if np.isnan(temp1).sum() > 0 or np.isinf(temp1).sum() > 0:
                            print('strange value in h_n_i')
                        if np.isnan(temp2).sum() > 0 or np.isinf(temp2).sum() > 0:
                            print('strange value in qhist')
        H = temp1
        Q = temp2

        # A --- Solving l0/l2 regularized problem
        qvec = Q.flatten(order='C')
        hvec = H.flatten(order='C')

        if delta > 0:  # delta控制0正则化的强度
            l0l2sol = self.L0L2sqlog_plus_linear(qvec=qvec, hvec=hvec,
                                                 delta=delta, nu=nu_alpha, omega=omega)
            avec = l0l2sol.get('x')
            l0norm = l0l2sol.get('l0norm')
        else:  # 等于0时没有0正则化
            avec = (-hvec + np.sqrt(hvec ** 2 + 4. * qvec * nu_alpha)) / (2. * nu_alpha)
            l0norm = (avec != 0).sum()
            l0l2sol = {'l0norm': l0norm, 'x': avec}

        alpha[:, :, :] = avec.reshape(alpha.shape, order='C')

        return alpha, l0norm, l0l2sol

    # 0正则化下求解alpha最优解
    def L0L2sqlog_plus_linear(self, qvec, hvec, delta, nu, omega):
        # Computing alpha_bar

        alpha_bar = (-hvec + np.sqrt(hvec ** 2 + 4. * nu * qvec)) / (2 * nu)

        if np.isnan(hvec).sum() > 0 or np.isinf(hvec).sum() > 0:
            print('strange value in hvec')

        if np.isnan(qvec).sum() > 0 or np.isinf(qvec).sum() > 0:
            print('strange value in qvec')

        if np.isnan(alpha_bar).sum() > 0 or np.isinf(alpha_bar).sum() > 0:
            print('strange value in alpha_bar')

        alpha_bar[alpha_bar < 0] = 0  ##### ADDED ######
        x = alpha_bar.copy()

        # 找到大于0的解
        mask0 = (alpha_bar != 0)
        alpha_bar1 = x[mask0]  # 只留下大于0的解
        x1 = alpha_bar1.copy()

        qvec1 = qvec[mask0]
        hvec1 = hvec[mask0]

        Phi_alpha_bar = qvec1 * np.log(alpha_bar1) - hvec1 * alpha_bar1 - (1 / 2) * nu * (alpha_bar1 ** 2)
        if np.isnan(Phi_alpha_bar).sum() > 0 or np.isinf(Phi_alpha_bar).sum() > 0:
            print('L0L2sqlog_plus_linear: strange value in log')

        Phi_omega = qvec1 * np.log(omega) - hvec1 * omega - (1 / 2) * nu * (omega ** 2)
        gain_to_off = Phi_omega - Phi_alpha_bar + delta

        # Checking if
        mask = (alpha_bar1 >= omega) & (gain_to_off > 0)
        x1[mask] = omega

        x[mask0] = x1

        l0norm = np.sum(x > omega)
        x_sparse = x.copy()
        x_sparse[x <= omega] = 0

        # 统计一下h中为0的数量
        zero_count1 = (x != 0).sum()
        zero_count2 = (x_sparse != 0).sum()

        obj = {'l0norm': l0norm, 'x': x, 'x_sparse': x_sparse, 'x_noL0': alpha_bar}
        return obj


    '''
    更新beta
    '''

    def update_beta(self,
                    train_events,
                    stations,
                    trains,
                    Delta,
                    histories,
                    histories_hop,
                    qself,
                    qhist,
                    timestamps,
                    alpha,
                    beta,
                    nu_beta,
                    A,
                    days,
                    day_list,
                    day_events,
                    ddtt):

        for v in range(len(beta)):
            v_events = train_events[v]
            B_beta_v = 0.0
            C_beta_v = (1 - qself[v_events]).sum()  # 当一个列车的所有晚点事件都是初始晚点事件时，该值为1
            # C_beta_v = 0
            # for tt in v_events:
            #     C_beta_v += qhist[tt, ~np.isnan(qhist[tt, :])].sum()
            for n in v_events:  # 遍历所有s_站点下发生的事件的id
                # if len(histories[n]) == 0:
                #     continue

                if n == 0:
                    continue

                '''
                加快计算方式
                '''
                # 定位该事件在day_events中的位置
                # day = days[n]
                # day_id = day_list.tolist().index(day)
                #
                # in_day_idx = day_events[day_id].tolist().index(n)
                # if in_day_idx == 0:
                #     continue
                #
                # last_event_index = day_events[day_id][in_day_idx - 1]

                # last_event_timestamp = max(timestamps[histories[n]])

                aa = qhist[n][histories[n]] * (timestamps[n] - timestamps[histories[n]])
                bb = alpha[histories_hop[n], v, trains[histories[n]]] * A[histories_hop[n], stations[n], stations[histories[n]]]

                Delta_last = Delta[n] - ddtt[n]
                cc = self.beta_derivative(Delta[n], beta[v]) - \
                        self.beta_derivative(Delta_last, beta[v])

                # cc = self.beta_derivative(Delta[n], beta[v]) - \
                #     self.beta_derivative(last_event_timestamp - timestamps[histories[n]], beta[v])

                dd = aa.sum() + (bb * cc).sum()

                B_beta_v += dd

            beta[v] = (-B_beta_v +
                       np.sqrt(B_beta_v ** 2
                               + 4. * nu_beta * C_beta_v)) / (2. * nu_beta)
            # 让beta_增加，应该减小nu_beta
            # if math.isinf(beta[v]) or math.isnan(beta[v]) or beta[v] < 0 or beta[v] > 10:   # beta等于0的时候是该列车都是初始晚点的时候
            #     raise ValueError("beta[v] out of range", beta[v])

        return beta

    '''
    更新q
    '''

    def update_q(self,
                 train_events,
                 timestamps,
                 trains,
                 stations,
                 Delta,
                 histories,
                 histories_hop,
                 beta,
                 alpha,
                 A,
                 qself,
                 qhist,
                 mu):

        N = len(timestamps)

        for n in range(N):  # n=0,...,N-1表达当前事件下标
            # print(n)
            mu_n = mu[trains[n]]

            lambda_n = mu_n

            '''
            快速计算方式
            '''
            qhist[n, histories[n]] = self.trigger(beta[trains[n]],
                                                  Delta[n],
                                                  alpha[histories_hop[n], trains[n], trains[histories[n]]],
                                                  A[histories_hop[n], stations[n], stations[histories[n]]])

            # print("q:", qhist[n, histories[n]])

            if np.isnan(qhist[n, histories[n]]).any() or np.isinf(qhist[n, histories[n]]).any():
                raise ValueError("lambda_i is nan")

            lambda_n += qhist[n, histories[n]].sum()

            # if lambda_n <= 0:
            #     raise ValueError("lambda_n is zero or negative, out of range: ", lambda_n)

            # if trains[n] == 1:
            #     print("qhist:", qhist[n, histories[n]], ", qself:", qself[n],
            #           ", beta:", beta[trains[n]], ", alpha:",
            #           alpha[histories_hop[n], trains[n], trains[histories[n]]],
            #           ", mu:", mu_n)


            qself[n] = mu_n / lambda_n
            qhist[n, histories[n]] = qhist[n, histories[n]] / lambda_n


        # v_q = (1 - qself[train_events[1]]).sum()
        # if v_q <= 0:
        #     raise ValueError("v_q is zero or negative, out of range: ", v_q)   # 因为alpha太小了

        return qself, qhist

    '''
    验证参数输入范围
    '''

    def verify_input(self, nu_mu, nu_beta, nu_alpha, eta):

        L2strengths = [nu_mu, nu_beta, nu_alpha]
        if eta < 0.5 or eta >= 1:
            raise ValueError('f{eta}:sparse_level must be in [0.5,1)')
        elif any(para <= 0 for para in L2strengths):
            raise ValueError('L2 strengths must be positive')


    '''
    找出某事件的历史事件，alpha不一定有关系
    '''
    def history_of(self, A_norm, n, t, s, k, Th, timestamps, stations, trains):
        events = []
        for index in range(n - 1, -1, -1):
            adj_s = stations[index]
            adj_t = timestamps[index]
            xx = A_norm[k, s, adj_s]
            bb = t - adj_t
            if xx > 0 and adj_t < t and bb <= Th:
                events.append(index)
                train_id = trains[index]
                print("和历史事件的时间差：", t - adj_t)
        print(n, " ", k, "-th neighbor history events: ", events)
        return events


    def initialize_mu(self, timestamps, train_events, a_mu, b_mu):
        T = timestamps[-1] - timestamps[0]
        D = len(train_events)
        mu = np.zeros(D)
        for k in range(D):
            Nk = train_events[k].shape[0]
            mu[k] = (Nk + a_mu - 1) / (T + b_mu)
        return mu


    '''
    因果推理和参数学习
    '''

    def learn(self,
              result_prefix_path,
              itr_max,
              reporting_interval,
              K,
              err_threshold,
              timestamps,
              trains,
              stations,
              days,
              A,
              a_mu,
              b_mu):

        '''
        保证事件按照时间顺序排序
        '''
        timestamps, trains, stations = \
            self.verify_timeorder(timestamps,
                                  trains,
                                  stations,
                                  self.Th)

        '''
        判断一些参数输入范围是否正确
        '''
        self.verify_input(self.nu_mu, self.nu_beta, self.nu_alpha, self.eta)

        '''
        获取V的集合以及N_V
        '''
        train_list = np.unique(trains)
        N_V = len(train_list)

        # '''
        # 获取S的集合以及N_S
        # '''
        station_list = np.unique(stations)

        '''
        获取day的集合以及N_day
        '''
        day_list = np.unique(days)
        N_day = len(day_list)

        '''
        获取事件总数
        '''
        N = len(timestamps)

        '''
        给出delta的值，alpha的0正则化强度常数
        '''
        delta = np.log(self.eta / (1 - self.eta))
        print('---- N(#events)={}, N_V(#trains)={}, Th(hist.len)={}'.format(N, N_V, self.Th))

        '''
        将某个列车的事件提取出来放在一起
        '''
        train_events = []  # train_events[k]则为k列车的事件id
        zero2N = np.arange(0, len(trains))
        for train in train_list:
            mask = (trains == train)
            train_events.append(zero2N[mask])

        '''
        将某个站点的事件提取出来放在一起
        '''
        station_events = {}
        zero2N = np.arange(0, len(stations))
        for station in station_list:
            mask = (stations == station)
            station_events[station] = zero2N[mask]

        '''
        将某天的事件提取出来放在一起
        '''
        day_events = []
        zero2N = np.arange(0, len(days))
        for day in day_list:
            mask = (days == day)
            day_events.append(zero2N[mask])

        '''
        获取每个事件的k阶相邻站点中的历史事件(n,k)，存到csv文件中减少每次训练的时间
        '''
        if not os.path.exists(result_prefix_path + 'histories_hop_' + str(K) + '_' + str(self.Th) + '.json'):
            histories = []  # 存储所有的邻居事件
            histories_hop = []  # 存储每个邻居事件对应的阶数
            for n in range(N):
                t = timestamps[n]
                s = stations[n]
                v = trains[n]
                if v == 0:
                    aa = v
                his_n = []
                hop_n = []
                for k in range(K + 1):
                    histories_of_k = self.history_of(A, n, t, s, k, self.Th, timestamps, stations, trains)
                    k_histories_trains = trains[histories_of_k]
                    his_n.extend(histories_of_k)
                    hop_n.extend([k for _ in range(len(histories_of_k))])
                histories.append(his_n)
                histories_hop.append(hop_n)
            with open(result_prefix_path + 'histories_' + str(self.Th) + '.json', 'w') as json_file:
                json.dump(histories, json_file, indent=4)
            with open(result_prefix_path + 'histories_hop_' + str(K) + '_' + str(self.Th) + '.json', 'w') as json_file:
                json.dump(histories_hop, json_file, indent=4)
        else:
            with open(result_prefix_path + 'histories_' + str(self.Th) + '.json', 'r') as json_file:
                histories = json.load(json_file)
            with open(result_prefix_path + 'histories_hop_' + str(K) + '_' + str(self.Th) + '.json', 'r') as json_file:
                histories_hop = json.load(json_file)

        '''
        计算每个事件和其历史事件的时间差（考虑了空间拓扑下的邻居）
        '''
        Delta = dict()
        for n in range(N):
            Delta[n] = timestamps[n] - timestamps[histories[n]]

        '''
        统计每个事件和其上一个事件的时间差
        '''
        ddtt = np.zeros([N])
        # ddtt: t(n) - t(n-1) for n = 1,..., N
        ddtt[0] = np.nan
        for n in range(1, N):
            i = 1
            if i > 1:
                while timestamps[n] == timestamps[n - i]:
                    i += 1
            ddtt[n] = timestamps[n] - timestamps[n - i]
            if ddtt[n] < 0:
                raise ValueError("ddtt is zero ornegative, out of range: ", ddtt)

        '''
        初始化参数
        '''
        # 初始化mu
        # mu = self.initialize_mu(timestamps, train_events, a_mu, b_mu)
        mu = np.random.uniform(*(0.0001, 0.001), N_V)

        # 初始化不同拓扑下的影响因子,alpha应该有地理位置的约束
        '''
        判断alpha错误是否学习到了K阶范围内不相邻站点的V之间的关系
        '''
        if not os.path.exists(result_prefix_path + 'topo_v.npy'):
            based_on_graph = np.zeros([K + 1, N_V, N_V])  # 根据G_S得到的基础G_V（alpha在G_V基础上变换）
            for k in range(K + 1):
                for s_1 in range(A.shape[1]):
                    for s_2 in range(A.shape[2]):
                        if A[k, s_1, s_2] == 1:
                            if s_1 not in station_events.keys() or s_2 not in station_events.keys():
                                continue
                            s_1_events = station_events[s_1]
                            s_2_events = station_events[s_2]
                            v_1_list = trains[s_1_events]
                            v_2_list = trains[s_2_events]
                            for a in v_1_list:
                                for b in v_2_list:
                                    based_on_graph[k][b][a] = based_on_graph[k][a][b] = 1
            np.save(result_prefix_path + 'topo_v.npy', based_on_graph)
        else:
            based_on_graph = np.load(result_prefix_path + 'topo_v.npy')

        alpha = np.ones([K + 1, N_V, N_V])
        # alpha = alpha * based_on_graph    # 强制掩码符合拓扑约束

        # alpha = np.random.uniform(*(0, 1), [K+1, N_V, N_V])

        # 初始化时间衰减因子
        # beta = mu.copy()
        beta = np.random.uniform(*(0.0001, 0.001), N_V)

        # 初始化q
        qself, qhist = self.initialize_q(mu,
                                        timestamps,
                                         stations,
                                         trains,
                                         Delta,
                                         histories,
                                         histories_hop,
                                         beta,
                                         alpha,
                                         A)

        '''
        MM迭代训练
        '''
        print('---- itr_max={},residual threshold={}'.format(itr_max, err_threshold))
        digits = 1 + int(np.abs(np.log10(err_threshold)))  # for showing progress

        Q = np.zeros([K + 1, N_V, N_V])
        H = np.zeros([K + 1, N_V, N_V])

        loglik = list()

        # 要更新的参数先保存一下
        mu_old = 2 * mu.copy()
        alpha_old = 2 * alpha.copy()
        beta_old = 2 * beta.copy()

        for itr in range(itr_max):
            # 更新动态环境影响参数
            mu = self.update_mu(timestamps,
                                qself,
                                self.nu_mu,
                                mu,
                                train_events,
                                days,
                                day_list,
                                day_events,
                                histories,
                                ddtt)

            # 更新列车影响因子
            alpha, l0norm, l0l2sol = self.update_alpha(alpha,
                                                       train_events,
                                                       stations,
                                                       trains,
                                                       timestamps,
                                                       histories,
                                                       histories_hop,
                                                       beta,
                                                       A,
                                                       Delta,
                                                       qhist,
                                                       H,
                                                       Q,
                                                       delta,
                                                       self.nu_alpha,
                                                       self.omega,
                                                       days,
                                                       day_list,
                                                       day_events,
                                                       ddtt,
                                                       based_on_graph,
                                                       K)

            # 更新时间衰减参数
            beta = self.update_beta(train_events,
                                    stations,
                                    trains,
                                    Delta,
                                    histories,
                                    histories_hop,
                                    qself,
                                    qhist,
                                    timestamps,
                                    alpha,
                                    beta,
                                    self.nu_beta,
                                    A,
                                    days,
                                    day_list,
                                    day_events,
                                    ddtt)

            # 更新实例级因果
            qself, qhist = self.update_q(train_events,
                                         timestamps,
                                         trains,
                                         stations,
                                         Delta,
                                         histories,
                                         histories_hop,
                                         beta,
                                         alpha,
                                         A,
                                         qself,
                                         qhist,
                                         mu)

            '''
            计算对数似然
            '''
            loglik0 = 0
            loglik1 = 0
            for n in range(N):  # n=0,...,N-1表达当前事件下标
                mu_n = mu[trains[n]]
                lambda_n = mu_n

                # if len(histories[n]) == 0:
                #     loglik0 += np.log(lambda_n)
                #     loglik1 -= 0
                #     continue

                if n == 0:
                    loglik0 += np.log(lambda_n)
                    loglik1 -= 0
                    continue

                alpha_k_v_v_i = alpha[histories_hop[n], trains[n], trains[histories[n]]]
                A_k_s_s_i = A[histories_hop[n], stations[n], stations[histories[n]]]
                lambda_i = self.trigger(beta[trains[n]], Delta[n], alpha_k_v_v_i, A_k_s_s_i)
                lambda_n += lambda_i.sum()

                loglik0 = loglik0 + np.log(lambda_n)

                # if math.isinf(loglik0) or math.isnan(loglik0):
                #     xx = np.log(lambda_n)
                #     raise ValueError("lambda_n is out of range: ", xx)

                # if n == 0:
                #     continue
                # else:
                #     # 定位该事件在day_events中的位置
                #     day = days[n]
                #     day_id = day_list.tolist().index(day)
                #     in_day_idx = day_events[day_id].tolist().index(n)
                #     if in_day_idx == 0:  # 如果是当天的第一个事件，默认没有上一个事件
                #         continue
                #     last_event_idx = day_events[day_id][in_day_idx - 1]
                #     delta_t = timestamps[n] - timestamps[last_event_idx]
                #     cc = self.temporal_decay_integral(Delta[n], beta[trains[n]]) - \
                #          self.temporal_decay_integral(timestamps[last_event_idx] - timestamps[histories[n]], beta[trains[n]])
                #     h = (alpha_k_v_v_i * A_k_s_s_i * cc).sum()
                    # print(f"last event in this day id is {last_event_idx}, and the delta_t is {delta_t}")

                # last_event_timestamp = max(timestamps[histories[n]])
                # delta_t = timestamps[n] - last_event_timestamp

                Delta_last = Delta[n] - ddtt[n]

                cc = self.temporal_decay_integral(Delta[n], beta[trains[n]]) - \
                     self.temporal_decay_integral(Delta_last, beta[trains[n]])

                # cc = self.temporal_decay_integral(Delta[n], beta[trains[n]]) - \
                #       self.temporal_decay_integral(last_event_timestamp - timestamps[histories[n]],
                #                                    beta[trains[n]])
                h = (alpha_k_v_v_i * A_k_s_s_i * cc).sum()

                loglik1 -= (mu_n * ddtt[n] + h)

            ln_Gauss_mu = - 0.5 * self.nu_mu * (mu * mu).sum()

            ln_Gauss_beta = - 0.5 * self.nu_beta * (beta * beta).sum()

            ln_Gauss_alpha = -0.5 * self.nu_alpha * (alpha * alpha).sum()

            ln_Bernoulli_alpha = - delta * l0norm

            loglik_reg = ln_Gauss_mu + ln_Gauss_beta + \
                         + ln_Gauss_alpha + ln_Bernoulli_alpha

            loglik_total = loglik0 + loglik1 + loglik_reg

            if loglik_total > 0:
                raise ValueError("loglik out of range: ", loglik_total)

            if len(loglik) > 1 and loglik_total < loglik[-1]:  # early stopping
                break

            loglik.append(loglik_total)

            # 检查收敛性
            err_mu = np.abs(1 - (mu * mu_old).sum() / np.sum(mu ** 2))
            err_beta = np.abs(1 - (beta * beta_old).sum() / np.sum(beta ** 2))
            err_alpha = np.abs(1 - (alpha * alpha_old).sum() / np.sum(alpha ** 2))

            print('---- itr={}'.format(itr + 1))
            print('\tresidual(mu, beta, alpha)=({:.4g},{:.4g},{:.4g}),loglik={}' \
                  .format(err_mu, err_beta, err_alpha, loglik_total))

            if err_mu < err_threshold and err_beta < err_threshold and \
                    err_alpha < err_threshold:
                print('---- Converged(th={}) at itr={}'.format(err_threshold, itr + 1))
                print('\tfinal residual(mu, beta, alpha)=({:.4g},{:.4g},{:.4g}),loglik={}' \
                      .format(err_mu, err_beta, err_alpha, loglik_total))
                break
            elif (itr + 1) % reporting_interval == 0 and itr != 0:
                print('{:4d}:residual(mu, beta, alpha)=('.format(itr + 1), end='')
                print('{:{dd}.{digits}f},'.format(err_mu, dd=digits + 2, digits=digits), end='')
                print('{:{dd}.{digits}f},'.format(err_beta, dd=digits + 2, digits=digits), end='')
                print('{:{dd}.{digits}f}'.format(err_alpha, dd=digits + 2, digits=digits), end='')
                print('),loglik={}'.format(loglik_total))

            mu_old = mu[:]
            alpha_old[:, :, :] = alpha[:, :, :]
            beta_old[:] = beta[:]

        # ===== 迭代结束 =====
        print('\teta={}(delta={:.4g}), nu_alpha={}, omega={}'. \
              format(self.eta, delta, self.nu_alpha, self.omega))
        print('\tnu_mu={}, nu_beta={}, final loglik={}'. \
              format(self.nu_mu, self.nu_beta, loglik_total))
        if delta > 0:  # 有0正则化
            # alpha_temp = l0l2sol.get('x_sparse').reshape([N_V, N_V], order='C')
            # alpha_final = np.ones([K+1, N_V, N_V]) * alpha_temp
            alpha_final = l0l2sol.get('x_sparse').reshape([K + 1, N_V, N_V], order='C')
        else:  # 无0正则化
            alpha_final = alpha.copy()

        for k in range(K + 1):
            for a in range(N_V):
                for b in range(N_V):
                    if based_on_graph[k][a][b] == 0 and alpha_final[k][a][b] != 0:
                        raise ValueError("不该学习到的怎么学到了", k, a, b)

        regularizer_params = {'eta': self.eta, 'delta': delta,
                              'nu_alpha': self.nu_alpha, 'omega': self.omega, 'nu_mu': self.nu_mu,
                              'nu_beta': self.nu_beta, 'Th': self.Th}
        learned_params = {'mu': mu,
                          'beta': beta, 'alpha': alpha_final,
                          'l0norm': l0norm,
                          'qself': qself, 'qhist': qhist,
                          'l0l2sol': l0l2sol, 'loglik': loglik}
        obj = {'learned_params': learned_params,
               'regularizer_params': regularizer_params}

        return obj

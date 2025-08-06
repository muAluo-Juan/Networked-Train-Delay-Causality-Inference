import json
import igraph as ig
from igraph.drawing.matplotlib.graph import mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from pypinyin import pinyin, Style
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import seaborn as sns
import networkx as nx
import matplotlib.patches as mpatches


'''
绘制列车因果图
'''
def draw_train_causal(file_path, out_path):
    G_V = np.load(file_path + 'G_V.npy')
    A = np.load(file_path + 'A.npy')   #  记录了不同阶数的站点
    events = pd.read_csv(file_path + 'events.csv')

    K = 2
    based_on_graph = np.zeros([K+1, len(np.array(G_V)[0]), len(np.array(G_V)[1])])  # 根据G_S得到的基础G_V
    G_V_topo = pd.DataFrame(G_V.copy()).replace(1, -1) # -1表示存在因果
    G_V_topo = np.array(G_V_topo)

    for k in range(K + 1):
        s_adj = A[k]
        for i in range(len(s_adj)):
            for j in range(len(s_adj[i])):
                if s_adj[i][j] == 1:  # 如果两个站点相邻，可以认为在这两个站点的两个列车也相邻
                    # 找到i站点上的列车
                    i_trains = events[events['station_id'] == i]['train_id'].values
                    # 找到j站点上的列车
                    j_trains = events[events['station_id'] == j]['train_id'].values
                    for a in i_trains:
                        for b in j_trains:
                            based_on_graph[k][b][a] = based_on_graph[k][a][b] = 1
                            if G_V[a][b] == 1:  # 存在因果关系，且具有拓扑邻居关系，就从-1转为1
                                G_V_topo[a][b] = 1
                            elif G_V[b][a] == 1:
                                G_V_topo[b][a] = 1

    G_V_topo_df = pd.DataFrame(G_V_topo)
    trains = json.load(open(file_path + 'trains.json', 'r')).keys()
    G_V_topo_df.columns = list(trains)
    G_V_topo_df.to_csv(out_path + 'G_V_topo.csv', index=False)

    np.save(out_path + 'V_topo_graph.npy', based_on_graph)

    plt.figure(figsize=(6, 6))
    plt.imshow(G_V, cmap='Reds', interpolation='none')
    plt.grid(False)
    plt.title('train casual graph')
    plt.xlabel('train')
    plt.ylabel('train')
    plt.savefig(out_path + 'G_V.pdf')
    plt.show()

def draw_train_causal_other(file_path, out_path):
    G_V = pd.read_csv(out_path + 'G_V.csv')
    v_graph = np.load(file_path + 'V_topo_graph.npy')  # 记录了不同阶数的站点

    G_V_topo = pd.DataFrame(G_V.copy()).replace(1, -1)  # 首先默认列车的站点都不相邻
    G_V_topo = np.array(G_V_topo)

    G_V = np.array(G_V)
    K = 2
    for i in range(len(G_V)):
        for j in range(len(G_V[i])):
            if G_V[i][j] == 1:  # 推理得到的列车相邻
                for k in range(K + 1):
                    s_adj = v_graph[k][i][j]
                    if s_adj == 1:
                        G_V_topo[i][j] = 1  # 为1表明推理得到的列车所在站点也相邻

    G_V_topo_df = pd.DataFrame(G_V_topo)  # 只展示其中50列车
    trains = json.load(open(file_path + 'trains.json', 'r')).keys()
    G_V_topo_df.columns = list(trains)
    G_V_topo_df.to_csv(out_path + 'G_V_topo.csv', index=False)

    plt.figure(figsize=(6, 6))
    plt.imshow(G_V, cmap='Reds', interpolation='none')
    plt.grid(False)
    plt.title('train casual graph')
    plt.xlabel('train')
    plt.ylabel('train')
    plt.savefig(out_path + 'G_V.pdf')
    plt.show()


'''
绘制晚点事件因果图
'''
def get_g_x_heatmap_data(ground_truth_path, infer_path):
    events = pd.read_csv(ground_truth_path + 'events.csv')
    # dates = np.unique(events['date'])[0:7]

    '''SingleL'''
    # dates = ['2019-10-08', '2019-10-09']
    dates = ['2019-10-10']
    target_train_id = [1]

    '''
    crossL
    '''
    # dates = ['2020-01-03', '2020-01-04']
    # target_train_id = [28]

    date_event_id = []   # 存储不同日期对应的事件
    for i in dates:
        date_event_id.append(events[events['date'] == i]['event_id'].values)

    target_train_events = []
    target_event_timestamps = []
    for i in dates:
        temp = events[(events['date'] == i) & (events['train_id'].isin(target_train_id))]['event_id'].values
        target_train_events.append(temp[0])   # 存储最后/第一个事件作为目标事件
        timestamp_target_event = events[events['event_id'] == temp[0]]['event_time'].values[0]
        temp = pd.DataFrame(temp)
        temp.to_csv(ground_truth_path + f'{i}_target_events.csv', index=False)
        target_event_timestamps.append(timestamp_target_event)

    ground_truth = np.load(ground_truth_path + 'G_X.npy')
    date_ground_truth = []
    for i in range(len(dates)):
        day_events_id = date_event_id[i]
        day_causal = ground_truth[day_events_id[0]: day_events_id[-1] + 1, day_events_id[0]: day_events_id[-1] + 1]
        date_ground_truth.append(day_causal)
        day_causal = pd.DataFrame(day_causal)
        day_causal.index = day_events_id
        day_causal.columns = day_events_id
        day_causal.to_csv(ground_truth_path + f'{dates[i]}_G_X.csv', index=True)
        target_event_causal = ground_truth[target_train_events[i], :]  # 找到它的原因事件
        causal_events_id = np.where(target_event_causal == 1)[0]
        if len(causal_events_id) < 1:
            causal_events_id = [target_train_events[i]]
        causal_events_timestamps = events[events['event_id'].isin(causal_events_id)]['event_time'].values
        delta_t = [target_event_timestamps[i] - causal_events_timestamps]
        delta_t = pd.DataFrame(delta_t)
        delta_t.to_csv(ground_truth_path + f'{dates[i]}_delta_t.csv', header=causal_events_id, index=False)


    CRHG = np.array(pd.read_csv(infer_path + 'CRHG/G_X.csv'))
    date_CRHG = []
    for i in range(len(dates)):
        day_events_id = date_event_id[i]
        day_causal = CRHG[day_events_id[0]: day_events_id[-1] + 1, day_events_id[0]: day_events_id[-1] + 1]
        date_CRHG.append(day_causal)
        day_causal = pd.DataFrame(day_causal)
        day_causal.index = day_events_id
        day_causal.columns = day_events_id
        day_causal.to_csv(infer_path + 'CRHG/' + f'{dates[i]}_G_X.csv', index=True)
        target_event_causal = CRHG[target_train_events[i], :]  # 找到它的原因事件
        causal_events_id = np.where(target_event_causal == 1)[0]
        if len(causal_events_id) < 1:
            causal_events_id = [target_train_events[i]]
        causal_events_timestamps = events[events['event_id'].isin(causal_events_id)]['event_time'].values
        delta_t = [target_event_timestamps[i] - causal_events_timestamps]
        delta_t = pd.DataFrame(delta_t)
        delta_t.to_csv(infer_path + 'CRHG/' + f'{dates[i]}_delta_t.csv', header=causal_events_id, index=False)

    ISAHP = np.load(infer_path + 'ISAHP/G_X.npy')
    date_ISAHP = []
    for i in range(len(dates)):
        day_events_id = date_event_id[i]
        day_causal = ISAHP[day_events_id[0]: day_events_id[-1] + 1, day_events_id[0]: day_events_id[-1] + 1]
        date_ISAHP.append(day_causal)
        day_causal = pd.DataFrame(day_causal)
        day_causal.index = day_events_id
        day_causal.columns = day_events_id
        day_causal.to_csv(infer_path + 'ISAHP/' + f'{dates[i]}_G_X.csv', index=True)
        target_event_causal = ISAHP[target_train_events[i], :]  # 找到它的原因事件
        causal_events_id = np.where(target_event_causal == 1)[0]
        if len(causal_events_id) < 1:
            causal_events_id = [target_train_events[i]]
        causal_events_timestamps = events[events['event_id'].isin(causal_events_id)]['event_time'].values
        delta_t = [target_event_timestamps[i] - causal_events_timestamps]
        delta_t = pd.DataFrame(delta_t)
        delta_t.to_csv(infer_path + 'ISAHP/' + f'{dates[i]}_delta_t.csv', header=causal_events_id, index=False)

    TDCausal = np.array(pd.read_csv(infer_path + 'tdcausal_gridsearch/G_X.csv'))
    date_TDCausal = []
    for i in range(len(dates)):
        day_events_id = date_event_id[i]
        day_causal = TDCausal[day_events_id[0]: day_events_id[-1] + 1, day_events_id[0]: day_events_id[-1] + 1]
        date_TDCausal.append(day_causal)
        day_causal = pd.DataFrame(day_causal)
        day_causal.index = day_events_id
        day_causal.columns = day_events_id
        day_causal.to_csv(infer_path + 'tdcausal_gridsearch/' + f'{dates[i]}_G_X.csv', index=True)
        target_event_causal = TDCausal[target_train_events[i], :]  # 找到它的原因事件
        causal_events_id = np.where(target_event_causal == 1)[0]
        if len(causal_events_id) < 1:
            causal_events_id = [target_train_events[i]]
        causal_events_timestamps = events[events['event_id'].isin(causal_events_id)]['event_time'].values
        delta_t = [target_event_timestamps[i] - causal_events_timestamps]
        delta_t = pd.DataFrame(delta_t)
        delta_t.to_csv(infer_path + 'tdcausal_gridsearch/' + f'{dates[i]}_delta_t.csv', header=causal_events_id, index=False)



'''
绘制图模式的因果图
'''
def draw_train_causal_graph(infer_path,
                            net_all_train_ids,
                            net_train_to_color,
                            train_num):
    # ===================== 获取绘图所需数据 ====================
    # G_V = np.array(pd.read_csv(infer_path + 'G_V.csv'))
    G_V = np.load(infer_path + 'G_V.npy')
    net_train_causal = []
    for i in range(len(G_V)):
        for j in range(len(G_V[i])):
            if G_V[i][j] == 1 and (i == train_num or j == train_num) and i in net_all_train_ids and j in net_all_train_ids:
               color_index_i = net_all_train_ids.tolist().index(i)
               color_index_j = net_all_train_ids.tolist().index(j)
               if (color_index_j, color_index_i) not in net_train_causal:
                   net_train_causal.append((color_index_j, color_index_i))  # 和列车名的下标对应上


    # =========================== 设定图的一些参数 ============================
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['ytick.labelsize'] = 18  # 纵坐标刻度字体大小
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))  # 绘制一行三列子图，不写1,3则是一个图

    # =========================== 绘制列车因果 ================================
    net_train_g = ig.Graph(net_train_causal, directed=True)
    # 定义列车颜色映射
    net_train_g.vs['color'] = [mcolors.hex2color(i) for i in net_train_to_color.values()]

    ig.plot(
        net_train_g,
        target=ax,
        edge_width=1,
        seed=42,
        vertex_size=0.3,
        # layout="circular",
        edge_arrow_size=0.01,  # 增大箭头
    )
    plt.tight_layout()
    plt.savefig(infer_path + '{}_train_causal_graph.pdf'.format(train_num))
    plt.show()

'''
处理绘制传播路线的数据（带经纬度）
'''
def process_propagation_graph():
    # singleL_causal = np.load('../synthetic_data/2019-10-08_11-07/nhp_jh_4/G_X.npy')
    # crossL_causal = np.load('../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/G_X.npy')
    net_causal = np.load('../real_data/2025-4-2_4-23/lz/G_X.npy')
    # net_causal = np.array(pd.read_csv('../result/real/2025-4-2_4-23/lz/ISAHP/G_X.csv'))
    # net_causal = np.load('../result/real/2025-4-2_4-23/lz/ISAHP/G_X.npy')
    # net_causal = np.array(pd.read_csv('../result/real/2025-4-2_4-23/lz/tdcausal_gridsearch/G_X.csv'))

    # 将存在因果关系的事件提取出来
    # singleL_events = pd.read_csv('../synthetic_data/2019-10-08_11-07/nhp_jh_4/events.csv')
    # singleL_events = singleL_events[(singleL_events['day'] == '2019-10-12')]

    # crossL_events = pd.read_csv('../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/events.csv')
    # crossL_events = crossL_events[(crossL_events['day'] == '2019-12-25')]

    net_events = pd.read_csv('../real_data/2025-4-2_4-23/lz/events.csv')

    date = '2025_4_8'

    net_events = net_events[(net_events['date'] == '2025/4/8')]  # 样例,只能用来筛选某一天的事件

    '''
    筛选出因果图中出现的事件
    '''
    # indices_1 = np.where(singleL_causal == 1)
    # indices_2 = np.where(crossL_causal == 1)
    indices_3 = np.where(net_causal == 1)
    # singleL_event_id = set()
    # crossL_event_id = set()
    net_event_id = set()
    # causal_csv_singleL = []
    # for i in range(len(indices_1[0])):
    #     source = indices_1[1][i]
    #     target = indices_1[0][i]
    #     if source in singleL_events['event_id'].values and target in singleL_events['event_id'].values:
    #         causal_csv_singleL.append([source, target])
    #         singleL_event_id.add(source)
    #         singleL_event_id.add(target)
    #
    # causal_csv_crossL = []
    # for i in range(len(indices_2[0])):
    #     source = indices_2[1][i]
    #     target = indices_2[0][i]
    #     if source in crossL_events['event_id'].values and target in crossL_events['event_id'].values:
    #         causal_csv_crossL.append([source, target])
    #         crossL_event_id.add(source)
    #         crossL_event_id.add(target)

    causal_csv_net = []
    for i in range(len(indices_3[0])):
        source = indices_3[1][i]
        target = indices_3[0][i]
        if source in net_events['event_id'].values and target in net_events['event_id'].values:
            causal_csv_net.append([source, target])
            net_event_id.add(source)
            net_event_id.add(target)

    # causal_csv_singleL = pd.DataFrame(causal_csv_singleL, columns=['source', 'target'])
    # causal_csv_crossL = pd.DataFrame(causal_csv_crossL, columns=['source', 'target'])
    # causal_csv_net = pd.DataFrame(causal_csv_net, columns=['source', 'target'])
    # causal_csv_singleL.to_csv('../synthetic_data/2019-10-08_11-07/nhp_jh_4/causal_path.csv', index=False)
    # causal_csv_crossL.to_csv('../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/causal_path.csv', index=False)
    # causal_csv_net.to_csv('../real_data/2025-4-2_4-23/lz/causal_path.csv', index=False)


    '''
    筛选出在微观因果路径中出现的事件
    '''
    # singleL_tri_events = singleL_events[singleL_events['event_id'].isin(list(singleL_event_id))]
    # crossL_tri_events = crossL_events[crossL_events['event_id'].isin(list(crossL_event_id))]
    net_tri_events = net_events[net_events['event_id'].isin(list(net_event_id))]

    def get_pinyin(name):
        result = pinyin(name, style=Style.NORMAL)
        formatted = ''.join([word[0] for word in result])
        formatted = formatted[0].capitalize() + formatted[1:]
        return f"{formatted} Railway Station"

    # 为每个事件标上经纬度
    # jh = pd.read_csv('../raw_data/jh.csv', encoding='gbk')  # 中文转一下英文
    # jh['station'] = jh['station'].apply(lambda x: get_pinyin(x))
    # jg = pd.read_csv('../raw_data/jg.csv', encoding='gbk')
    # jg['station'] = jg['station'].apply(lambda x: get_pinyin(x))
    # hk = pd.read_csv('../raw_data/hk.csv', encoding='gbk')
    # hk['station'] = hk['station'].apply(lambda x: get_pinyin(x))
    # xl = pd.read_csv('../raw_data/xl.csv', encoding='gbk')
    # xl['station'] = xl['station'].apply(lambda x: get_pinyin(x))

    # 交叉綫上使用的京滬
    # cross_jh = jh.copy()

    dx2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/dx2025.csv', encoding='gbk')
    jg2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/jg2025.csv', encoding='gbk')
    jh2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/jh2025.csv', encoding='gbk')
    lx2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/lx2025.csv', encoding='gbk')
    xl2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/xl2025.csv', encoding='gbk')
    yx2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/yx2025.csv', encoding='gbk')
    jq2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/jq2025.csv', encoding='gbk')
    shiji2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/shiji2025.csv', encoding='gbk')
    xc2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/xc2025.csv', encoding='gbk')
    hf2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/hf2025.csv', encoding='utf8')
    lanzhang2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/lanzhang2025.csv', encoding='gbk')
    lz2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/lz2025.csv', encoding='gbk')
    na2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/na2025.csv', encoding='gbk')
    nh2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/nh2025.csv', encoding='gbk')
    hst2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/hst2025.csv', encoding='gbk')
    qrcj2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/qrcj2025.csv', encoding='gbk')
    qy2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/qy2025.csv', encoding='gbk')
    zccj2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/zccj2025.csv', encoding='gbk')
    cg2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/cg2025.csv', encoding='utf8')
    shh2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/shh2025.csv', encoding='gbk')
    hh2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/hh2025.csv', encoding='gbk')
    hncj2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/hncj2025.csv', encoding='utf8')
    hsh2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/hsh2025.csv', encoding='utf8')
    jbcj2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/jbcj2025.csv', encoding='utf8')
    jj2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/jj2025.csv', encoding='utf8')
    jz2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/jz2025.csv', encoding='utf8')
    ln2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/ln2025.csv', encoding='utf8')
    cjh2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/cjh2025.csv', encoding='gbk')
    xvlian2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/xvlian2025.csv', encoding='gbk')
    xy2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/xy2025.csv', encoding='gbk')
    yt2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/yt2025.csv', encoding='gbk')
    zt2025 = pd.read_csv('../real_data/2025-4-2_4-23/lz/lines/zt2025.csv', encoding='gbk')


    # 站点和编号对应上
    def change_name_to_number(name, station_pairs):
        aa = station_pairs.keys()
        if name not in station_pairs.keys():
            return -1       # 可能有未篩選到的站點
        return station_pairs[name]
    singleL_pairs = json.load(open('../synthetic_data/2019-10-08_11-07/nhp_jh_4/stations.json', 'r'))
    crossL_pairs = json.load(open('../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/stations.json', 'r'))
    net_pairs = json.load(open('../real_data/2025-4-2_4-23/lz/stations.json', 'r', encoding='utf-8'))

    # jh['station_id'] = jh['station'].apply(lambda x: change_name_to_number(x, singleL_pairs))
    #
    # cross_jh['station_id'] = cross_jh['station'].apply(lambda x: change_name_to_number(x, crossL_pairs))
    # jg['station_id'] = jg['station'].apply(lambda x: change_name_to_number(x, crossL_pairs))
    # hk['station_id'] = hk['station'].apply(lambda x: change_name_to_number(x, crossL_pairs))
    # xl['station_id'] = xl['station'].apply(lambda x: change_name_to_number(x, crossL_pairs))

    dx2025['station_id'] = dx2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    jg2025['station_id'] = jg2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    jh2025['station_id'] = jh2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    lx2025['station_id'] = lx2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    xl2025['station_id'] = xl2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    yx2025['station_id'] = yx2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    jq2025['station_id'] = jq2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    shiji2025['station_id'] = shiji2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    xc2025['station_id'] = xc2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    hf2025['station_id'] = hf2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    lanzhang2025['station_id'] = lanzhang2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    lz2025['station_id'] = lz2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    na2025['station_id'] = na2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    nh2025['station_id'] = nh2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    hst2025['station_id'] = hst2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    qrcj2025['station_id'] = qrcj2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    qy2025['station_id'] = qy2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    zccj2025['station_id'] = zccj2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    cg2025['station_id'] = cg2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    shh2025['station_id'] = shh2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    hh2025['station_id'] = hh2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    hncj2025['station_id'] = hncj2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    hsh2025['station_id'] = hsh2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    jbcj2025['station_id'] = jbcj2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    jj2025['station_id'] = jj2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    jz2025['station_id'] = jz2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    ln2025['station_id'] = ln2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    cjh2025['station_id'] = cjh2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    xvlian2025['station_id'] = xvlian2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    xy2025['station_id'] = xy2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    yt2025['station_id'] = yt2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))
    zt2025['station_id'] = zt2025['station'].apply(lambda x: change_name_to_number(x, net_pairs))


    # 将站点id和经纬度对应
    def get_lon_lat(array, longitude_dict, latitude_dict):
        station_id = array[-1]
        if station_id == -1:
            return
        longitude = array[2]
        latitude = array[3]
        if station_id not in longitude_dict:
            longitude_dict[station_id] = longitude
            latitude_dict[station_id] = latitude

    # singleL_longitude = dict()
    # singleL_latitude = dict()
    # crossL_longitude = dict()
    # crossL_latitude = dict()
    net_longitude = dict()
    net_latitude = dict()

    # for i in np.array(jh):
    #     get_lon_lat(i, singleL_longitude, singleL_latitude)
    # for i in np.array(cross_jh):
    #     get_lon_lat(i, crossL_longitude, crossL_latitude)
    # for i in np.array(jg):
    #     get_lon_lat(i, crossL_longitude, crossL_latitude)
    # for i in np.array(hk):
    #     get_lon_lat(i, crossL_longitude, crossL_latitude)
    # for i in np.array(xl):
    #     get_lon_lat(i, crossL_longitude, crossL_latitude)
    for i in np.array(dx2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(jg2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(jh2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(lx2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(xl2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(yx2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(jq2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(shiji2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(xc2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(hf2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(lanzhang2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(lz2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(na2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(nh2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(hst2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(qrcj2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(qy2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(zccj2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(cg2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(shh2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(hh2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(hncj2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(hsh2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(jbcj2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(jj2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(jz2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(ln2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(cjh2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(xvlian2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(xy2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(yt2025):
        get_lon_lat(i, net_longitude, net_latitude)
    for i in np.array(zt2025):
        get_lon_lat(i, net_longitude, net_latitude)

    # 爲每個事件添加经纬度
    def add_longitude(station_id, longitude_dict):
        return longitude_dict[station_id]
    def add_latitude(station_id, latitude_dict):
        return latitude_dict[station_id]
    # singleL_tri_events['longitude'] = singleL_tri_events['station_id'].apply(lambda x: add_longitude(x, singleL_longitude))
    # singleL_tri_events['latitude'] = singleL_tri_events['station_id'].apply(lambda x: add_latitude(x, singleL_latitude))
    # crossL_tri_events['longitude'] = crossL_tri_events['station_id'].apply(lambda x: add_longitude(x, crossL_longitude))
    # crossL_tri_events['latitude'] = crossL_tri_events['station_id'].apply(lambda x: add_latitude(x, crossL_latitude))
    net_tri_events['longitude'] = net_tri_events['station_id'].apply(lambda x: add_longitude(x, net_longitude))
    net_tri_events['latitude'] = net_tri_events['station_id'].apply(lambda x: add_latitude(x, net_latitude))

    # singleL_tri_events.to_csv('../synthetic_data/2019-10-08_11-07/nhp_jh_4/tri_events.csv', index=False)
    #
    # crossL_tri_events.to_csv('../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/tri_events.csv', index=False)

    net_tri_events.to_csv('../real_data/2025-4-2_4-23/lz/{}_tri_events.csv'.format(date), index=False)
    # net_tri_events.to_csv('../result/real/2025-4-2_4-23/lz/tdcausal_gridsearch/causal_path_for_all_trains/{}_tri_events.csv'.format(date),
    #                       index=False)
    # net_tri_events.to_csv(
    #     '../result/real/2025-4-2_4-23/lz/tdcausal_gridsearch/{}_tri_events.csv'.format(date),
    #                           index=False)

    '''
    为causal直接带上经纬度
    '''
    final_causal_net = []
    for i in range(len(causal_csv_net)):
        source = causal_csv_net[i][0]
        target = causal_csv_net[i][1]
        source_longitude = net_tri_events[net_tri_events['event_id'] == source]['longitude'].values[0]
        target_longitude = net_tri_events[net_tri_events['event_id'] == target]['longitude'].values[0]
        source_train = net_tri_events[net_tri_events['event_id'] == source]['train_id'].values[0]
        source_station = net_tri_events[net_tri_events['event_id'] == source]['station_id'].values[0]
        source_latitude = net_tri_events[net_tri_events['event_id'] == source]['latitude'].values[0]
        target_latitude = net_tri_events[net_tri_events['event_id'] == target]['latitude'].values[0]
        target_train = net_tri_events[net_tri_events['event_id'] == target]['train_id'].values[0]
        target_station = net_tri_events[net_tri_events['event_id'] == target]['station_id'].values[0]
        final_causal_net.append([source, source_train, source_station, source_longitude, source_latitude,
                                 target, target_train, target_station, target_longitude, target_latitude])

    # causal_csv_singleL = pd.DataFrame(causal_csv_singleL, columns=['source', 'target'])
    # causal_csv_crossL = pd.DataFrame(causal_csv_crossL, columns=['source', 'target'])
    final_causal_net = pd.DataFrame(final_causal_net, columns=['source', 'source_train', 'source_station', 'source_longitude', 'source_latitude',
                                                               'target', 'target_train', 'target_station', 'target_longitude', 'target_latitude'])
    # causal_csv_singleL.to_csv('../synthetic_data/2019-10-08_11-07/nhp_jh_4/causal_path.csv', index=False)
    # causal_csv_crossL.to_csv('../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/causal_path.csv', index=False)
    # final_causal_net.to_csv('../result/real/2025-4-2_4-23/lz/ISAHP/causal_path.csv', index=False)
    # final_causal_net.to_csv('../result/real/2025-4-2_4-23/lz/tdcausal_gridsearch/{}_causal_path.csv'.format(date), index=False)
    final_causal_net.to_csv('../real_data/2025-4-2_4-23/lz/{}_causal_path.csv'.format(date), index=False)

    '''
    进一步提取出连续的传播路径=================================================
    '''
    # # 构建有向图
    # G = nx.DiGraph()
    # for _, row in final_causal_net.iterrows():
    #     source = str(row["source"])
    #     target = str(row["target"])
    #     G.add_edge(source, target)
    #
    # # 识别源头节点（入度为0）和终止节点（出度为0）
    # sources = [node for node in G.nodes if G.in_degree(node) == 0]
    # sinks = [node for node in G.nodes if G.out_degree(node) == 0]
    #
    # # 提取所有完整链路（无分支路径）
    # all_chains = []
    # for src in sources:
    #     for snk in sinks:
    #         try:
    #             # 获取所有无环路径
    #             chains = nx.all_simple_paths(G, source=src, target=snk)
    #             for chain in chains:
    #                 # 处理分支节点：若路径中某节点出度>1，则拆分为独立链路
    #                 if any(G.out_degree(node) > 1 for node in chain):
    #                     temp_chain = []
    #                     for node in chain:
    #                         temp_chain.append(node)
    #                         if G.out_degree(node) > 1:
    #                             all_chains.append(temp_chain.copy())
    #                             temp_chain = [temp_chain[0]]  # 分支后以该节点为新起点
    #                 else:
    #                     all_chains.append(chain)
    #         except nx.NetworkXNoPath:
    #             continue
    #
    # all_chains = set(tuple(chain) for chain in all_chains)
    #
    # # 按链路长度分类存储到TXT文件
    # chain_lengths = {}
    # for chain in all_chains:
    #     length = len(chain)
    #     if length not in chain_lengths:
    #         chain_lengths[length] = []
    #     chain_lengths[length].append(chain)
    #
    # # 写入TXT文件
    # for length, chains in chain_lengths.items():
    #     if length < 3:
    #         continue
    #     filename = f"../result/real/2025-4-2_4-23/lz/tdcausal_gridsearch/causal_path_for_all_trains/{date}_chains.txt"
    #     with open(filename, "a") as f:
    #         for chain in chains:
    #             f.write(",".join(chain) + "\n")
    #     print(f"已存储 {len(chains)} 条 {length}节点链路到 {filename}")
    #
    # print(f"共处理 {len(all_chains)} 条完整链路")


'''
处理目标列车的传播路线数据（带经纬度）
'''
def process_target_propagation_graph(path):
    date = '2025_4_8'
    day_tri_events = pd.read_csv(path + '{}_tri_events.csv'.format(date))
    day_causal_path = pd.read_csv(path + '{}_causal_path.csv'.format(date))

    target_train_id = 20    # 2025/4/8-20, 2025/4/12-46, 7、11、26
    # 找到原因事件和目标事件是目标列车的事件
    target_events_causal = day_causal_path[(day_causal_path['source_train'] == target_train_id) |
                                    (day_causal_path['target_train'] == target_train_id)]
    target_events_causal.to_csv(path + '{}_{}_causal_path.csv'.format(date, target_train_id), index=False)
    target_source_events = target_events_causal['source'].values
    target_des_events = target_events_causal['target'].values
    target_events = day_tri_events[(day_tri_events['event_id'].isin(target_source_events)) | (day_tri_events['event_id'].isin(target_des_events))]
    target_events.to_csv(path + '{}_{}_tri_events.csv'.format(date, target_train_id), index=False)

'''
获得数据集绘图数据
'''
def get_propagation_data(event_path,
                         causal,
                         statin_pairs_path,
                         train_pairs_path,
                         target_train_id,
                         target_date,
                         train_causal_path):

    np.random.seed(42)

    # 列车运行数据（列车, 站点, 时间戳）
    events = pd.read_csv(event_path)
    train_pairs = json.load(open(train_pairs_path, 'r', encoding='utf-8'))
    station_pairs = json.load(open(statin_pairs_path, 'r', encoding='utf-8'))
    target_events = events[(events["train_id"].isin(target_train_id)) & (events["date"] == target_date)]
    target_events_id = target_events['event_id'].values
    target_station_id = target_events['station_id'].values  # 用于标记本列车运营线路
    train_causal = np.load(train_causal_path)

    # 站点因果事件
    causal_edges = []  # 提取出目标事件的相关因果边
    causal_events_id = []  # 提取出目标事件的相关因果事件
    for i in np.array(causal):
        source = i[0]
        target = i[1]
        if source in target_events_id:
            causal_edges.append((source, target))
            causal_events_id.append(target)
        elif target in target_events_id:
            causal_edges.append((source, target))
            causal_events_id.append(source)

    all_events_id = np.unique(list(target_events_id) + list(causal_events_id))
    all_events = np.array(events[events['event_id'].isin(all_events_id)])
    all_events_station_id = all_events[:, 2]
    all_events_train_id = all_events[:, 1]
    all_events_station_names = []

    def get_pinyin(name):
        result = pinyin(name, style=Style.NORMAL)
        formatted = ''.join([word[0] for word in result])
        formatted = formatted[0].capitalize() + formatted[1:]
        return f"{formatted}"

    for i in all_events_station_id:
        for key, value in station_pairs.items():
            if value == i:
                all_events_station_names.append(get_pinyin(key))
    all_events_train_names = []
    for i in all_events_train_id:
        for key, value in train_pairs.items():
            if value == i:
                all_events_train_names.append(key)

    all_trains_id = pd.unique(all_events_train_id)
    all_train_names = []
    for i in all_trains_id:
        for key, value in train_pairs.items():
            if value == i:
                all_train_names.append(key)
    all_stations_id = pd.unique(all_events_station_id)
    all_station_names = []
    target_station_names = []
    for i in all_stations_id:
        for key, value in station_pairs.items():
            if value == i:
                all_station_names.append(get_pinyin(key))
                if i in target_station_id:
                    target_station_names.append(get_pinyin(key))

    all_trains_causal = []
    for i in range(len(train_causal)):
        for j in range(len(train_causal[i])):
            if train_causal[i][j] == 1 and i in all_trains_id and j in all_trains_id:
                index_j = all_trains_id.tolist().index(j)
                index_i = all_trains_id.tolist().index(i)
                if (index_j, index_i) not in all_trains_causal:
                    all_trains_causal.append((index_j, index_i))

    return all_station_names, target_station_names, all_train_names, \
    all_events, all_events_station_names, all_events_train_names,\
           all_events_id, causal_edges, all_trains_causal

'''
贝塞尔曲线，绘制传播路线（不带经纬度）
'''
def draw_causal_graph_for_dataset():
    # ===================== 获取绘图所需数据 ====================
    singleL_causal_csv = np.load('../synthetic_data/2019-10-08_11-07/nhp_jh_4/G_X.npy')
    crossL_causal_csv = np.load('../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/G_X.npy')
    net_causal_csv = np.load('../real_data/2025-4-2_4-23/lz/G_X.npy')

    # 将存在因果关系的事件提取出来
    singleL_events = pd.read_csv('../synthetic_data/2019-10-08_11-07/nhp_jh_4/events.csv')
    crossL_events = pd.read_csv('../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/events.csv')
    net_events = pd.read_csv('../real_data/2025-4-2_4-23/lz/events.csv')

    '''
    筛选出因果图中出现的事件
    '''
    indices_1 = np.where(singleL_causal_csv == 1)
    indices_2 = np.where(crossL_causal_csv == 1)
    indices_3 = np.where(net_causal_csv == 1)
    singleL_event_id = set()
    crossL_event_id = set()
    net_event_id = set()
    causal_singleL = []
    for i in range(len(indices_1[0])):
        source = indices_1[1][i]
        target = indices_1[0][i]
        if source in singleL_events['event_id'].values and target in singleL_events['event_id'].values:
            causal_singleL.append([source, target])
            singleL_event_id.add(source)
            singleL_event_id.add(target)

    causal_crossL = []
    for i in range(len(indices_2[0])):
        source = indices_2[1][i]
        target = indices_2[0][i]
        if source in crossL_events['event_id'].values and target in crossL_events['event_id'].values:
            causal_crossL.append([source, target])
            crossL_event_id.add(source)
            crossL_event_id.add(target)

    causal_net = []
    for i in range(len(indices_3[0])):
        source = indices_3[1][i]
        target = indices_3[0][i]
        if source in net_events['event_id'].values and target in net_events['event_id'].values:
            causal_net.append([source, target])
            net_event_id.add(source)
            net_event_id.add(target)

    singleL_event_path = '../synthetic_data/2019-10-08_11-07/nhp_jh_4/events.csv'
    singleL_statin_pairs_path = '../synthetic_data/2019-10-08_11-07/nhp_jh_4/stations.json'
    singleL_train_pairs_path = '../synthetic_data/2019-10-08_11-07/nhp_jh_4/trains.json'
    singleL_target_train_id = [1]  #[5]  # [1] # [10]
    singleL_target_date = '2019-10-08'  # '2019-10-08'
    singleL_train_causal_path = '../synthetic_data/2019-10-08_11-07/nhp_jh_4/G_V.npy'

    crossL_event_path = '../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/events.csv'
    crossL_statin_pairs_path = '../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/stations.json'
    crossL_train_pairs_path = '../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/trains.json'
    crossL_target_train_id = [28] #[28, 65]  # [77 ,69, 60, 65, 61,38,11]
    crossL_target_date = '2020-01-04'#'2019-12-30'
    crossL_train_causal_path = '../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/G_V.npy'

    net_event_path = '../real_data/2025-4-2_4-23/lz/events.csv'
    net_statin_pairs_path = '../real_data/2025-4-2_4-23/lz/stations.json'
    net_train_pairs_path = '../real_data/2025-4-2_4-23/lz/trains.json'
    net_target_train_id = [7]  # [20] #[24]
    net_target_date = '2025/4/8'  # '2025/4/8' #'2025/4/8'
    net_train_causal_path = '../real_data/2025-4-2_4-23/lz/G_V.npy'

    singleL_all_station_names, \
    singleL_target_station_names, \
    singleL_all_train_names, \
    singleL_all_events, \
    singleL_all_events_station_names, \
    singleL_all_events_train_names, \
    singleL_all_events_id, \
    singleL_causal_edges,\
    singleL_train_causal = \
        get_propagation_data(singleL_event_path,
                                 causal_singleL,
                                 singleL_statin_pairs_path,
                                 singleL_train_pairs_path,
                                 singleL_target_train_id,
                                 singleL_target_date,
                                 singleL_train_causal_path)

    crossL_all_station_names, \
    crossL_target_station_names, \
    crossL_all_train_names, \
    crossL_all_events, \
    crossL_all_events_station_names, \
    crossL_all_events_train_names, \
    crossL_all_events_id, \
    crossL_causal_edges,\
    crossL_train_causal= \
        get_propagation_data(crossL_event_path,
                                 causal_crossL,
                                 crossL_statin_pairs_path,
                                 crossL_train_pairs_path,
                                 crossL_target_train_id,
                                 crossL_target_date,
                                 crossL_train_causal_path)

    net_all_station_names, \
    target_station_names, \
    net_all_train_names, \
    net_all_events, \
    net_all_events_station_names, \
    net_all_events_train_names,\
    net_all_events_id, \
    net_causal_edges,\
    net_train_causal = \
        get_propagation_data(net_event_path,
                                 causal_net,
                                 net_statin_pairs_path,
                                 net_train_pairs_path,
                                 net_target_train_id,
                                 net_target_date,
                                 net_train_causal_path)

    # =========================== 设定图的一些参数 ============================
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['ytick.labelsize'] = 8  # 纵坐标刻度字体大小
    fig, ax = plt.subplots(2, 3, figsize=(15, 5))  # 绘制一行三列子图，不写1,3则是一个图

    # =========================== 绘制列车因果 ================================
    singleL_train_g = ig.Graph(len(singleL_all_train_names), singleL_train_causal, directed=True)
    crossL_train_g = ig.Graph(len(crossL_all_train_names), crossL_train_causal, directed=True)
    net_train_g = ig.Graph(len(net_all_train_names), net_train_causal, directed=True)
    # 绘制列车因果图
    # 定义列车颜色映射
    import colorsys
    def generate_nature_colors(num_colors):
        """
        生成符合Nature风格的配色（RGBA值在0-1范围内）
        参数:
            num_colors: 需要生成的颜色数量
        返回:
            list of RGBA元组，格式为 (R, G, B, A) 浮点数 (0-1范围)
        """
        np.random.seed(42)  # 固定种子保证可复现性

        # 基础Nature色相范围（蓝/绿/棕/灰）[3,4](@ref)
        base_hues = [0.55, 0.35, 0.08, 0.75]  # HSV色相值

        extended_hues = [
            # 冷色系扩展（蓝-绿区间）
            0.50, 0.45, 0.40, 0.30, 0.25,  # 蓝绿过渡色（5种）

            # 暖色系扩展（红-黄区间）
            0.02, 0.95, 0.90, 0.85,  # 深红/酒红/紫红（4种）
            0.12, 0.15, 0.18, 0.22,  # 橙黄/土黄/赭石（4种）

            # 中性色扩展
            0.65, 0.70, 0.80,  # 青灰/银灰/岩灰（3种）
            0.05, 0.00,  # 深棕/炭黑（2种）

            # 特殊高亮色
            0.92, 0.28, 0.62  # 品红/青柠/薰衣草紫（3种）
        ]

        # 完整色相列表（24个节点）
        full_nature_hues = base_hues + extended_hues

        colors = []
        for _ in range(num_colors):
            # 1. 选择基础色相
            base_hue = np.random.choice(full_nature_hues)

            # 2. 微调色相 (±0.05范围)
            hue = np.clip(base_hue + np.random.uniform(-0.5, 0.5), 0, 1)

            # 3. 控制饱和度 (0.3-0.6) 和明度 (0.4-0.7)
            saturation = np.random.uniform(0.3, 0.6)
            value = np.random.uniform(0.4, 0.7)

            # 4. 转换为0-1范围的RGB（关键修正）[9](@ref)
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            rgba = (r, g, b, 1.0)  # A通道固定为1.0（不透明）

            colors.append(rgba)

        return colors

    color = generate_nature_colors(len(crossL_all_train_names))

    target_color = (0.6941, 0.1333, 0.1333, 1.0)  # 目标车为红色

    singleL_train_to_color = {train: color[i] if train != 'G8912' else target_color for i, train in enumerate(singleL_all_train_names)}

    crossL_train_to_color = {train: color[i] if train != 'G6105' else target_color for i, train in enumerate(crossL_all_train_names)}

    net_train_to_color = {train: color[i] if train != 'G1168/5/8' else target_color for i, train in enumerate(net_all_train_names)}

    singleL_train_g.vs['color'] = [mcolors.hex2color(i) for i in singleL_train_to_color.values()]
    crossL_train_g.vs['color'] = [mcolors.hex2color(i) for i in crossL_train_to_color.values()]
    net_train_g.vs['color'] = [mcolors.hex2color(i) for i in net_train_to_color.values()]

    node_size = 0.4
    edge_width = 1.5
    ig.plot(
        singleL_train_g,
        # edge_curved=1,
        target=ax[0][0],
        edge_width=edge_width,
        # seed=42,
        # vertex_size=node_size,
    )

    ig.plot(
        crossL_train_g,
        target=ax[0][1],
        edge_width=edge_width,
        # seed=42,
        # vertex_size=node_size,
    )

    ig.plot(
        net_train_g,
        target=ax[0][2],
        edge_width=edge_width,
        # seed=42,
        # vertex_size=node_size,
    )
    ax[0][0].set_aspect('equal')
    ax[0][1].set_aspect('equal')
    ax[0][2].set_aspect('equal')

    # ============================= 绘制站点水平线 =============================
    singleL_station_to_y = {station: i for i, station in enumerate(singleL_all_station_names)}
    crossL_station_to_y = {station: i for i, station in enumerate(crossL_all_station_names)}
    net_station_to_y = {station: i for i, station in enumerate(net_all_station_names)}
    # 绘制站点水平线
    for station, y in singleL_station_to_y.items():
        ax[1][0].axhline(y, color="black" if station in singleL_target_station_names else 'silver', alpha=1, linestyle="--", zorder=-1)
    for station, y in crossL_station_to_y.items():
        ax[1][1].axhline(y, color="black" if station in crossL_target_station_names else 'silver', alpha=1, linestyle="--", zorder=-1)
    for station, y in net_station_to_y.items():
        ax[1][2].axhline(y, color="black" if station in target_station_names else 'silver', alpha=1, linestyle="--", zorder=-1)

    # 设置坐标轴标签
    ax[1][0].set_yticks(list(singleL_station_to_y.values()), fontsize=8)
    ax[1][0].set_yticklabels(list(i.split(" ")[0] for i in singleL_station_to_y.keys()))
    ax[1][0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax[1][1].set_yticks(list(crossL_station_to_y.values()), fontsize=8)
    ax[1][1].set_yticklabels(list(i.split(" ")[0] for i in crossL_station_to_y.keys()))
    ax[1][1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax[1][2].set_yticks(list(net_station_to_y.values()), fontsize=8)
    ax[1][2].set_yticklabels(list(net_station_to_y.keys()))
    ax[1][2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # ========================= 绘制事件节点 ========================
    # 使用虚拟时间戳
    def generate_mdates_times(num_events, interval_minutes=30):
        # 生成起始时间（默认从当前时间开始，可自定义）
        start_time = pd.Timestamp.now().floor('min')  # 示例起始时间（精确到分钟）
        # 生成时间序列（等间隔）
        times = pd.date_range(
            start=start_time,
            periods=num_events,
            freq=f"{interval_minutes}T"  # 间隔分钟数
        )

        # 转换为 mdates.date2num 可接受的数值时间戳
        return mdates.date2num(times)


    singleL_interval_minutes = 30
    singleL_x = generate_mdates_times(len(singleL_all_events), singleL_interval_minutes)
    singleL_y = [singleL_station_to_y[station] for station in singleL_all_events_station_names]
    singleL_node_colors = [singleL_train_to_color[train] for train in singleL_all_events_train_names]

    crossL_interval_minutes = 30
    crossL_x = generate_mdates_times(len(crossL_all_events), crossL_interval_minutes)
    crossL_y = [crossL_station_to_y[station] for station in crossL_all_events_station_names]
    crossL_node_colors = [crossL_train_to_color[train] for train in crossL_all_events_train_names]

    net_interval_minutes = 30
    net_x = generate_mdates_times(len(net_all_events), net_interval_minutes)
    net_y = [net_station_to_y[station] for station in net_all_events_station_names]
    net_node_colors = [net_train_to_color[train] for train in net_all_events_train_names]

    # 绘制散点（节点）
    ax[1][0].scatter(singleL_x, singleL_y, c=singleL_node_colors, s=50, edgecolor="black", zorder=2)
    ax[1][1].scatter(crossL_x, crossL_y, c=crossL_node_colors, s=50, edgecolor="black", zorder=2)
    ax[1][2].scatter(net_x, net_y, c=net_node_colors, s=50, edgecolor="black", zorder=2)

    # 设置坐标轴标签
    ax[1][0].set_xticks([])  # 移除横坐标刻度
    ax[1][0].set_xlabel("")  # 可选：移除横坐标标签
    ax[1][1].set_xticks([])  # 移除横坐标刻度
    ax[1][1].set_xlabel("")  # 可选：移除横坐标标签
    ax[1][2].set_xticks([])  # 移除横坐标刻度
    ax[1][2].set_xlabel("")  # 可选：移除横坐标标签

    # 添加事件标签
    # for i, row in enumerate(all_events):
    #     ax.text(x[i], y[i], row[0], ha="right", va="bottom", fontsize=8)

    # 添加图例
    # 创建图例句柄
    marker_size = 3
    singleL_legend_elements = [
        Line2D([0], [0],
               marker='o',
               color='w',
               label=train,
               markerfacecolor=color,
               markersize=marker_size,
               markeredgecolor='black')
        for train, color in zip(singleL_all_train_names, singleL_train_to_color.values())
    ]

    crossL_legend_elements = [
        Line2D([0], [0],
               marker='o',
               color='w',
               label=train,
               markerfacecolor=color,
               markersize=marker_size,
               markeredgecolor='black')
        for train, color in zip(crossL_all_train_names, crossL_train_to_color.values())
    ]

    net_legend_elements = [
        Line2D([0], [0],
               marker='o',
               color='w',
               label=train,
               markerfacecolor=color,
               markersize=marker_size,
               markeredgecolor='black')
        for train, color in zip(net_all_train_names, net_train_to_color.values())
    ]


    # 添加图例
    ax[1][0].legend(
        handles=singleL_legend_elements,
        # title="train number",
        loc='upper left',
        # bbox_to_anchor=(0, 0),
        ncol=1,
        prop={'size': 8},
        facecolor='white',  # 背景颜色
        edgecolor='silver',
        framealpha=1,  # 背景透明度（0~1）
        fancybox=True,  # 圆角边框（False为直角）
        # shadow=True  # 是否显示阴影
    )

    ax[1][1].legend(
        handles=crossL_legend_elements,
        # title="train number",
        loc='upper left',
        # bbox_to_anchor=(0, 0),
        ncol=1,
        prop={'size': 8},
        facecolor='white',  # 背景颜色
        edgecolor='silver',
        framealpha=1,  # 背景透明度（0~1）
        fancybox=True,  # 圆角边框（False为直角）
        # shadow=True  # 是否显示阴影
    )

    ax[1][2].legend(
        handles=net_legend_elements,
        loc='upper left',
        # bbox_to_anchor=(0, 0),
        ncol=1,
        prop={'size': 8},
        facecolor='white',  # 背景颜色
        edgecolor='silver',
        framealpha=1,  # 背景透明度（0~1）
        fancybox=True,  # 圆角边框（False为直角）
        # shadow=True  # 是否显示阴影
    )

    # ============ 绘制事件因果边 =============
    # 创建igraph图对象
    singleL_g = ig.Graph(n=len(singleL_all_events), edges=singleL_causal_edges, directed=True)
    crossL_g = ig.Graph(n=len(crossL_all_events), edges=crossL_causal_edges, directed=True)
    net_g = ig.Graph(n=len(net_all_events), edges=net_causal_edges, directed=True)

    # 计算边的起点和终点坐标（转换为matplotlib坐标）
    singleL_edge_coords = []
    crossL_edge_coords = []
    net_edge_coords = []
    for e in singleL_g.es:
        source, target = e.tuple
        source_idx = singleL_all_events_id.tolist().index(source)
        target_idx = singleL_all_events_id.tolist().index(target)
        sx, sy = singleL_x[source_idx], singleL_y[source_idx]
        tx, ty = singleL_x[target_idx], singleL_y[target_idx]
        singleL_edge_coords.append(((sx, sy), (tx, ty)))
    for e in crossL_g.es:
        source, target = e.tuple
        source_idx = crossL_all_events_id.tolist().index(source)
        target_idx = crossL_all_events_id.tolist().index(target)
        sx, sy = crossL_x[source_idx], crossL_y[source_idx]
        tx, ty = crossL_x[target_idx], crossL_y[target_idx]
        crossL_edge_coords.append(((sx, sy), (tx, ty)))
    for e in net_g.es:
        source, target = e.tuple
        source_idx = net_all_events_id.tolist().index(source)
        target_idx = net_all_events_id.tolist().index(target)
        sx, sy = net_x[source_idx], net_y[source_idx]
        tx, ty = net_x[target_idx], net_y[target_idx]
        net_edge_coords.append(((sx, sy), (tx, ty)))

    # 绘制因果边
    for (sx, sy), (tx, ty) in singleL_edge_coords:
        ax[1][0].annotate(
            "",
            xy=(tx, ty),
            xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>",
                # color="darkred",  # 更深的红色
                color="black",
                lw=1.5,  # 更粗的线宽
                alpha=1,  # 不透
                shrinkA=2,  # 箭头起点距离节点边缘 5 点
                shrinkB=2,  # 箭头终点距离节点边缘 5 点
                mutation_scale=12,  # 箭头头部尺寸
                zorder=1  # 图层在节点下方
            ),
            zorder=1)  # 确保箭头在底层

    for (sx, sy), (tx, ty) in crossL_edge_coords:
        ax[1][1].annotate(
            "",
            xy=(tx, ty),
            xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>",
                # color="darkred",  # 更深的红色
                color="black",
                lw=1,  # 更粗的线宽
                alpha=1,  # 不透
                shrinkA=2,  # 箭头起点距离节点边缘 5 点
                shrinkB=2,  # 箭头终点距离节点边缘 5 点
                mutation_scale=12,  # 箭头头部尺寸
                zorder=1  # 图层在节点下方
            ),
            zorder=1)  # 确保箭头在底层

    for (sx, sy), (tx, ty) in net_edge_coords:
        ax[1][2].annotate(
            "",
            xy=(tx, ty),
            xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>",
                # color="darkred",  # 更深的红色
                color="black",
                lw=1,  # 更粗的线宽
                alpha=1,  # 不透
                shrinkA=2,  # 箭头起点距离节点边缘 5 点
                shrinkB=2,  # 箭头终点距离节点边缘 5 点
                mutation_scale=12,  # 箭头头部尺寸
                zorder=1  # 图层在节点下方
            ),
            zorder=1)  # 确保箭头在底层

    # 调整子图间距（水平间距设为 0.3）
    # plt.subplots_adjust(wspace=0.3)
    plt.tight_layout(pad=1)  # pad 控制整体边距宽度（单位：英寸）
    plt.savefig('delay_propagation_path.pdf', dpi=300)
    plt.show()


def cal_propagation_recovery_rate(ground_path, infer_path):
    train = 11
    truth_g_v = np.array(pd.read_csv(ground_path + '{}_causal_path.csv'.format(train))[['source', 'target']])
    infer_g_v = np.array(pd.read_csv(infer_path + '{}_causal_path.csv'.format(train))[['source', 'target']])

    count = 0
    for i in range(truth_g_v.shape[0]):
        source, target = truth_g_v[i][0], truth_g_v[i][1]
        for j in range(infer_g_v.shape[0]):
            if source == infer_g_v[j][0] and target == infer_g_v[j][1]:
                count += 1
    print(count / len(truth_g_v))


# draw_train_causal('../synthetic_data/2019-10-08_11-07/nhp_jh_4/', '../synthetic_data/2019-10-08_11-07/nhp_jh_4/')
# draw_train_causal('../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/', '../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/')
# draw_train_causal('../real_data/2025-4-2_4-23/lz/', '../real_data/2025-4-2_4-23/lz/')
# draw_train_causal_other('../synthetic_data/2019-10-08_11-07/nhp_jh_4/', '../result/synthetic/2019-10-08_11-07/nhp_jh_4/tdcausal_gridsearch/')
# draw_train_causal_other('../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/', '../result/synthetic/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/CAUSE/')
# draw_train_causal_other('../real_data/2025-4-2_4-23/lz/', '../result/real/2025-4-2_4-23/lz/CRHG/')

# get_g_x_heatmap_data('../synthetic_data/2019-10-08_11-07/nhp_jh_4/', '../result/synthetic/2019-10-08_11-07/nhp_jh_4/')
# get_g_x_heatmap_data('../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/', '../result/synthetic/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/')


process_propagation_graph()
# process_target_propagation_graph('../result/real/2025-4-2_4-23/lz/tdcausal_gridsearch/')
process_target_propagation_graph('../real_data/2025-4-2_4-23/lz/')

# draw_causal_graph_for_dataset()

# cal_propagation_recovery_rate('../real_data/2025-4-2_4-23/lz/', '../result/real/2025-4-2_4-23/lz/tdcausal_gridsearch/')
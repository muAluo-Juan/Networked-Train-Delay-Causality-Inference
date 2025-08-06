import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, roc_curve, auc, mean_squared_error

'''
返回AUROC
'''
def cal_auroc(pred, true, name, file_path):
    auroc = roc_auc_score(true.flatten(), pred.flatten())  # ROC曲线下面积

    number_of_correctly_pre_edges = np.sum((pred == 1) & (true == 1))

    true_edges = np.sum((true == 1))
    TPR = number_of_correctly_pre_edges / true_edges

    number_of_pre_edges = np.sum((pred == 1) & (true == 0))
    false_edges = np.sum((true == 0))
    FPR = number_of_pre_edges / false_edges
    print("FPR = ", FPR, "TPR = ", TPR)

    fpr, tpr, threshold = roc_curve(true.flatten(), pred.flatten())
    roc_auc = auc(fpr, tpr)
    return auroc, fpr, tpr, roc_auc, FPR, TPR


'''
返回SHD（结构汉明距离）
'''
def cal_structural_hanmming_distance(pred, true):
    shd = np.sum(true != pred)
    return shd

'''
绘制NLL曲线
'''
def draw_nll(nll, name, file_path):
    x_raw = np.arange(len(nll))
    y_raw = np.array(nll)

    x_smooth = np.linspace(x_raw.min(), x_raw.max(), 500)
    spl = make_interp_spline(x_raw, y_raw, k=3)
    y_smooth = spl(x_smooth)

    plt.plot(x_raw, y_raw, 'o', alpha=0.3, label='Raw NLL')
    plt.plot(x_smooth, y_smooth, 'r-', lw=2, label='Smoothed Curve')
    plt.xlabel('Training Steps')
    plt.ylabel('Negative Log-Likelihood')
    plt.title(f'{name} Smoothed NLL Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path + name + ' NLL.pdf')
    plt.show()

'''
评估TDCausal
'''
def evaluate_TDCausal(result_prefix_path, generate_path_prefix):
    with open(result_prefix_path, 'rb') as f:
        result = pickle.load(f)

        '''
        将推理得到的因果图进行保存
        '''
        alpha = result['learned_params']['alpha']
        num_eval = 0
        G_V = np.zeros((alpha.shape[1], alpha.shape[2]))
        for k in range(alpha.shape[0]):
            for i in range(alpha.shape[1]):
                for j in range(alpha.shape[2]):
                    if alpha[k][i][j] > 0:
                        G_V[i][j] = 1
                        num_eval += 1
        print("推理出的G_V为：", G_V)
        G_V_pd = pd.DataFrame(G_V)
        G_V_pd.to_csv(result_prefix_path + 'G_V.csv', index=False)

        qhist = result['learned_params']['qhist']
        events = pd.read_csv(generate_path_prefix + 'events.csv')
        '''
        读取数据集的地面真值
        '''
        G_V_true = np.load(generate_path_prefix + 'G_V.npy')
        G_X_true = np.load(generate_path_prefix + 'G_X.npy')


        percent = np.count_nonzero(G_X_true) / len(G_X_true.flatten())


        qhist = np.where(np.isnan(qhist), 0, qhist)

        threshold = np.percentile(qhist.flatten(), (1 - (percent)) * 100)   # 0.025

        # threshold = 0

        G_X = np.where((qhist <= threshold), 0, 1)

        percent_infer_final = np.count_nonzero(G_X) / len(G_X.flatten())

        print("推理出的G_X为：", G_X)
        G_X_pd = pd.DataFrame(G_X)
        G_X_pd.to_csv(result_prefix_path + 'G_X.csv', index=False)

        '''
        计算或绘制指标结果
        '''
        G_V_auroc, G_V_fpr, G_V_tpr, G_V_roc_auc, G_V_FPR, G_V_TPR = cal_auroc(G_V, G_V_true, 'TDCausal', result_prefix_path)
        G_V_shd = cal_structural_hanmming_distance(G_V.flatten(), G_V_true.flatten())
        print("G_V TDCausal: auroc=", G_V_auroc, "fpr=", G_V_fpr, "tpr=", G_V_tpr, "shd=", G_V_shd)

        '''

        '''
        G_X_auroc, G_X_fpr, G_X_tpr, G_X_roc_auc, G_X_FPR, G_X_TPR = cal_auroc(G_X, G_X_true, 'TDCausal', result_prefix_path)
        G_X_shd = cal_structural_hanmming_distance(G_X.flatten(), G_X_true.flatten())

        # 打开文件并写入内容
        with open(result_prefix_path + "evaluate.txt", "w") as f:
            print(
                f"G_V TDCausal: FPR={G_V_FPR}, TPR={G_V_TPR}, auroc={G_V_auroc}, fpr={G_V_fpr}, "
                f"tpr={G_V_tpr}, shd={G_V_shd}"
                f"G_X TDCausal: FPR={G_X_FPR}, TPR={G_X_TPR}, auroc={G_X_auroc}, fpr={G_X_fpr}, "
                f"tpr={G_X_tpr}, shd={G_X_shd}", file=f)

        print("G_X TDCausal: auroc=", G_X_auroc, "fpr=", G_X_fpr, "tpr=", G_X_tpr, "shd=", G_X_shd)

        return G_V_TPR, G_X_TPR, G_V_FPR, G_X_FPR


'''
评估L0 Hawkes
'''
def evaluate_L0Hawkes(result_prefix_path, generate_path_prefix):
    with open(result_prefix_path, 'rb') as f:
        result = pickle.load(f)
        nll = result['learned_params']['loglik']

        '''
        将推理得到的因果图进行保存
        '''
        A = result['learned_params']['A']
        mask = (A != 0)
        indices = np.where(mask)
        G_V = np.zeros((len(A), len(A)))
        for i, j in zip(*indices):
            G_V[i][j] = 1

        print("推理出的G_V为：", G_V)

        G_V_pd = pd.DataFrame(G_V)
        G_V_pd.to_csv(result_prefix_path + 'G_V.csv', index=False)

        qhist = result['learned_params']['qhist']

        '''
        读取数据集的地面真值
        '''
        G_X_true = np.load(generate_path_prefix + 'G_X.npy')

        percent = np.count_nonzero(G_X_true) / len(G_X_true.flatten())

        qhist = np.where(np.isnan(qhist), 0, qhist)

        # threshold = np.percentile(qhist.flatten(), (1 - (percent)) * 100)  # 0.025

        threshold = 0

        mask = (~np.isnan(qhist)) & (qhist > threshold)
        indices = np.where(mask)
        G_X = np.zeros((len(qhist), len(qhist)))
        for i, j in zip(*indices):
            if i >= 240:
                G_X[i][i - j] = 1
            elif i > 0 and i < 240:
                G_X[i][j] = 1

        print("推理出的G_X为：", G_X)
        G_X_pd = pd.DataFrame(G_X)
        G_X_pd.to_csv(result_prefix_path + 'G_X.csv', index=False)


        '''
        读取数据集的地面真值
        '''
        G_V_true = np.load(generate_path_prefix + 'G_V.npy')
        G_X_true = np.load(generate_path_prefix + 'G_X.npy')

        # draw_nll(nll, 'L0Hawkes', result_prefix_path)

        '''
        计算或绘制指标结果
        '''
        G_V_auroc, G_V_fpr, G_V_tpr, G_V_roc_auc, G_V_FPR, G_V_TPR = cal_auroc(G_V.flatten(), G_V_true.flatten(),
                                                                               'CRHG', result_prefix_path)
        G_V_shd = cal_structural_hanmming_distance(G_V.flatten(), G_V_true.flatten())
        # 打开文件并写入内容
        print("G_V L0Hawkes: auroc=", G_V_auroc, "fpr=", G_V_fpr, "tpr=", G_V_tpr, "shd=", G_V_shd)


        G_X_auroc, G_X_fpr, G_X_tpr, G_X_roc_auc, G_X_FPR, G_X_TPR = cal_auroc(G_X.flatten(), G_X_true.flatten(),
                                                                               'CRHG', result_prefix_path)

        G_X_shd = cal_structural_hanmming_distance(G_X.flatten(), G_X_true.flatten())
        print("G_X L0Hawkes: auroc=", G_X_auroc, "fpr=", G_X_fpr, "tpr=", G_X_tpr, "shd=", G_X_shd)

        with open(result_prefix_path + "evaluate.txt", "w") as f:
            print(
                f"G_V CRHG: FPR={G_V_FPR}, TPR={G_V_TPR}, auroc={G_V_auroc}, fpr={G_V_fpr}, "
                f"tpr={G_V_tpr}, shd={G_V_shd}"
                f"G_X CRHG: FPR={G_X_FPR}, TPR={G_X_TPR}, auroc={G_X_auroc}, fpr={G_X_fpr}, "
                f"tpr={G_X_tpr}, shd={G_X_shd}", file=f)

        return G_V_TPR, G_X_TPR, G_V_FPR, G_X_FPR


'''
评估THP
'''
def evaluate_THP(result_prefix_path, generate_path_prefix):
    with open(result_prefix_path, 'rb') as f:
        result = pickle.load(f)

        '''
        将推理得到的因果图进行保存
        '''
        graph = result['graph']
        num_eval = 0
        G_V = np.zeros((graph.shape[0], graph.shape[1]))
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                if graph[i][j] > 0:
                    G_V[j][i] = 1
                    num_eval += 1
        print("推理出的G_V为：", G_V)
        # 绘制列车因果图
        plt.figure(figsize=(6, 6))
        plt.imshow(G_V, cmap='Blues', interpolation='none')
        plt.grid(False)
        plt.title('train casual graph')
        plt.xlabel('train')
        plt.ylabel('train')
        plt.savefig(result_prefix_path + 'G_V.pdf')
        plt.show()

        '''
        读取数据集的地面真值
        '''
        G_V_true = np.load(generate_path_prefix + 'G_V.npy')


        '''
        计算或绘制指标结果
        '''
        G_V_auroc, G_V_fpr, G_V_tpr, G_V_roc_auc, G_V_FPR, G_V_TPR = cal_auroc(G_V, G_V_true, 'THP', result_prefix_path)
        G_V_shd = cal_structural_hanmming_distance(G_V.flatten(), G_V_true.flatten())

        print("G_V THP: auroc=", G_V_auroc, "fpr=", G_V_fpr, "tpr=", G_V_tpr, "shd=", G_V_shd)

        # 打开文件并写入内容
        with open(result_prefix_path + "evaluate.txt", "w") as f:
            print(
                f"G_V THP: FPR={G_V_FPR}, TPR={G_V_TPR}, auroc={G_V_auroc}, fpr={G_V_fpr}, "
                f"tpr={G_V_tpr}, shd={G_V_shd}", file=f)

        return G_V_TPR, G_V_FPR


'''
评估MLE-SGL
'''
def evaluate_MLESGL(result_prefix_path, generate_path_prefix):
    '''
    获取推理得到的G_V
    '''
    result = loadmat(result_prefix_path)['result']
    G_V_value = result['A'][0][0]
    # threshold = 0.000005  # singleL
    threshold = 0    # net
    # threshold = 0.002  # crossL
    G_V = (G_V_value > threshold).astype(int)
    print("推理出的G_V为：", G_V)
    G_V_pd = pd.DataFrame(G_V)
    G_V_pd.to_csv(result_prefix_path + 'G_V.csv', index=False, header=False)

    # 绘制列车因果图
    plt.figure(figsize=(6, 6))
    plt.imshow(G_V, cmap='Blues', interpolation='none')
    plt.grid(False)
    plt.title('train casual graph')
    plt.xlabel('train')
    plt.ylabel('train')
    plt.savefig(result_prefix_path + 'G_V.pdf')
    plt.show()

    '''
    读取数据集的地面真值
    '''
    G_V_true = np.load(generate_path_prefix + 'G_V.npy')

    '''
    计算或绘制指标结果
    '''
    G_V_auroc, G_V_fpr, G_V_tpr, G_V_roc_auc, G_V_FPR, G_V_TPR = cal_auroc(G_V, G_V_true, 'MLE-SGL', result_prefix_path)
    G_V_shd = cal_structural_hanmming_distance(G_V.flatten(), G_V_true.flatten())
    print("G_V MLE_SGL: auroc=", G_V_auroc, "fpr=", G_V_fpr, "tpr=", G_V_tpr, "shd=", G_V_shd)

    # 打开文件并写入内容
    with open(result_prefix_path + "evaluate.txt", "w") as f:
        print(
            f"G_V MLE_SGL: FPR={G_V_FPR}, TPR={G_V_TPR}, auroc={G_V_auroc}, fpr={G_V_fpr}, "
            f"tpr={G_V_tpr}, shd={G_V_shd}", file=f)

    return G_V_TPR, G_V_FPR


'''
评估CAUSE
'''
def evaluate_CAUSE(result_prefix_path, generate_path_prefix):
    '''
    获取推理得到的G_V
    '''
    G_V = np.array(pd.read_csv(result_prefix_path))
    print("推理出的G_V为：", G_V)
    # 绘制列车因果图
    plt.figure(figsize=(6, 6))
    plt.imshow(G_V, cmap='Blues', interpolation='none')
    plt.grid(False)
    plt.title('train casual graph')
    plt.xlabel('train')
    plt.ylabel('train')
    plt.savefig(result_prefix_path + 'G_V.pdf')
    plt.show()

    '''
    读取数据集的地面真值
    '''
    G_V_true = np.load(generate_path_prefix + 'G_V.npy')

    '''
    计算或绘制指标结果
    '''
    G_V_auroc, G_V_fpr, G_V_tpr, G_V_roc_auc, G_V_FPR, G_V_TPR = cal_auroc(G_V, G_V_true, 'MLE-SGL', result_prefix_path)
    G_V_shd = cal_structural_hanmming_distance(G_V.flatten(), G_V_true.flatten())
    print("G_V CAUSE: auroc=", G_V_auroc, "fpr=", G_V_fpr, "tpr=", G_V_tpr, "shd=", G_V_shd)

    # 打开文件并写入内容
    with open(result_prefix_path + "evaluate.txt", "w") as f:
        print(
            f"G_V CAUSE: FPR={G_V_FPR}, TPR={G_V_TPR}, auroc={G_V_auroc}, fpr={G_V_fpr}, "
            f"tpr={G_V_tpr}, shd={G_V_shd}", file=f)

    return G_V_TPR, G_V_FPR


'''
评估cLSTM
'''
def evaluate_cLSTM(result_prefix_path, generate_path_prefix):
    '''
    获取推理得到的G_V
    '''
    G_V = np.array(pd.read_csv(result_prefix_path))
    print("推理出的G_V为：", G_V)
    # 绘制列车因果图
    plt.figure(figsize=(6, 6))
    plt.imshow(G_V, cmap='Blues', interpolation='none')
    plt.grid(False)
    plt.title('train casual graph')
    plt.xlabel('train')
    plt.ylabel('train')
    plt.savefig(result_prefix_path + 'G_V.pdf')
    plt.show()

    '''
    读取数据集的地面真值
    '''
    G_V_true = np.load(generate_path_prefix + 'G_V.npy')

    '''
    计算或绘制指标结果
    '''
    G_V_auroc, G_V_fpr, G_V_tpr, G_V_roc_auc, G_V_FPR, G_V_TPR = cal_auroc(G_V, G_V_true, 'MLE-SGL', result_prefix_path)
    G_V_shd = cal_structural_hanmming_distance(G_V.flatten(), G_V_true.flatten())
    print("G_V cLSTM: auroc=", G_V_auroc, "fpr=", G_V_fpr, "tpr=", G_V_tpr, "shd=", G_V_shd)

    # 打开文件并写入内容
    with open(result_prefix_path + "evaluate.txt", "w") as f:
        print(
            f"G_V cLSTM: FPR={G_V_FPR}, TPR={G_V_TPR}, auroc={G_V_auroc}, fpr={G_V_fpr}, "
            f"tpr={G_V_tpr}, shd={G_V_shd}", file=f)

    return G_V_TPR, G_V_FPR


def mape(predicted, actual):
    """
    计算 MAPE（平均绝对百分比误差）

    参数:
    actual: 实际值数组 (list/np.array)
    predicted: 预测值数组 (list/np.array)

    返回:
    MAPE 值（百分比形式）
    """
    # 转换为 NumPy 数组
    actual = np.array(actual)
    predicted = np.array(predicted)

    # 检查长度一致性
    if len(actual) != len(predicted):
        raise ValueError("实际值和预测值长度必须相同")

    # 检查实际值中是否有0（避免除零错误）
    if np.any(actual == 0):
        raise ValueError("实际值包含0，会导致除法错误")

    # 计算绝对百分比误差
    percentage_errors = np.abs((actual - predicted) / actual)

    # 计算平均值并转为百分比
    mape = np.mean(percentage_errors) * 100
    return mape


'''
统计不同类型晚点事件数量
'''
def cal_ini_and_tri_delay_events(ground_truth_path, infer_path):
    events = pd.read_csv(ground_truth_path + 'events.csv')
    dates = pd.unique(events['date'])
    G_X_true = np.load(ground_truth_path + 'G_X.npy')

    # CRHG、TDCausal
    G_X_infer = np.array(pd.read_csv(infer_path + 'params_33.pkG_X.csv'))

    # ISAHP
    # G_X_infer = np.load(infer_path + 'G_X.npy')

    infer_tri_events = []
    infer_ini_events = []
    for i in range(len(events)):
        if G_X_infer[i, :].sum() > 0:
            infer_tri_events.append(i)
        else:
            infer_ini_events.append(i)

    true_tri_events = []
    true_ini_events = []
    for i in range(len(events)):
        if G_X_true[i, :].sum() > 0:
            true_tri_events.append(i)
        else:
            true_ini_events.append(i)

    event_mape = []
    for date in dates:
        day_events = events[events['date'] == date]['event_id'].values
        count = 0
        for day_event in day_events:
            if day_event in infer_tri_events and day_event not in infer_ini_events:
                count += 1
        day_infer_tri_events = count
        day_infer_ini_events = len(day_events) - day_infer_tri_events

        count = 0
        for day_event in day_events:
            if day_event in true_tri_events and day_event not in true_ini_events:
                count += 1
        day_true_tri_events = count
        day_true_ini_events = len(day_events) - day_true_tri_events

        event_mape.append([date,
                           day_true_ini_events,
                           day_true_tri_events,
                           day_infer_ini_events,
                           day_infer_tri_events])

    res_pd = pd.DataFrame(event_mape)
    res_pd.columns = ['date',
                      'true_ini_events',
                      'true_tri_events',
                      'infer_ini_events',
                      'infer_tri_events']
    res_pd.to_csv(infer_path + '33event_number_infer_error.csv',
                    index=False)

    mape_tri = mape(res_pd['infer_tri_events'].values, res_pd['true_tri_events'].values)
    mape_ini = mape(res_pd['infer_ini_events'].values, res_pd['true_ini_events'].values)
    with open(infer_path + "33mape.txt", "w") as f:
        print(
            f"ini mape: {mape_ini}, "
            f"tri mape: {mape_tri}"
            , file=f)
    print("mape_ini: ", mape_ini)
    print("mape_tri: ", mape_tri)


'''
模型推理结果四个指标值计算
'''
# evaluate_TDCausal('../result/synthetic/2019-10-08_11-07/nhp_jh_4/tdcausal_gridsearch/' + 'params_0.pk', '../synthetic_data/2019-10-08_11-07/nhp_jh_4/')
# evaluate_TDCausal('../result/synthetic/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/tdcausal_gridsearch3/' + 'params_0.pk', '../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/')
# evaluate_TDCausal('../result/real/2025-4-2_4-23/lz/tdcausal_gridsearch/' + 'params_27.pk', '../real_data/2025-4-2_4-23/lz/')

# evaluate_L0Hawkes('../result/synthetic/2019-10-08_11-07/nhp_jh_4/CRHG/' + 'params.pk', '../synthetic_data/2019-10-08_11-07/nhp_jh_4/')
# evaluate_L0Hawkes('../result/synthetic/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/CRHG/' + 'params5.pk', '../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/')
# evaluate_L0Hawkes('../result/real/2025-4-2_4-23/lz/CRHG/' + 'params8.pk', '../real_data/2025-4-2_4-23/lz/')

# evaluate_THP('../result/synthetic/2019-10-08_11-07/nhp_jh_4/thp/' + 'params5.pk', '../synthetic_data/2019-10-08_11-07/nhp_jh_4/')

# evaluate_MLESGL('../result/real/2025-4-2_4-23/lz/MLE_SGL/' + 'params12.mat', '../real_data/2025-4-2_4-23/lz/')
# evaluate_MLESGL('../result/synthetic/2019-10-08_11-07/nhp_jh_4/MLE_SGL/' + 'params5.mat', '../synthetic_data/2019-10-08_11-07/nhp_jh_4/')
# evaluate_MLESGL('../result/synthetic/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/MLE_SGL/' + 'params4.mat', '../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/')

# evaluate_CAUSE('../result/synthetic/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/ISAHP/' + 'causal_matrix.csv', '../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/')
# evaluate_cLSTM('../result/synthetic/2019-10-08_11-07/nhp_jh_4/cLSTM/' + 'G_V_cLSTM.csv', '../synthetic_data/2019-10-08_11-07/nhp_jh_4/')


'''
统计推理出的每一天的初始晚点和连带晚点事件的数量和比例
'''
# cal_ini_and_tri_delay_events('../real_data/2025-4-2_4-23/lz/',
#                              '../result/real/2025-4-2_4-23/lz/tdcausal_gridsearch/')
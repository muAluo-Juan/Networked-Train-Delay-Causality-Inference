'''
网格搜索第四版本模型最优程序
'''
import pickle
from itertools import product
import numpy as np
import pandas as pd
from utils.evaluate import evaluate_TDCausal
from model.TDCausal import TDCausal


# 结果存储路径前缀
result_prefix_path = '../result/synthetic/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/tdcausal_gridsearch/'

# 实验数据集存储路径前缀
generate_path_prefix = '../synthetic_data/2019-11-08_01-07/nhp_jh_jg_xl_hk_150/'

# 模拟数据集

train_delay_data = pd.read_csv(generate_path_prefix + 'events.csv')

# 提取输入模型的数据
trains = np.array(train_delay_data['train_id'])
stations = np.array(train_delay_data['station_id'])
timestamps = np.array(train_delay_data['event_time'])
days = np.array(train_delay_data['date'])
A = np.load(generate_path_prefix + 'A.npy')

# 参数网格
param_grid = {
    'nu_mu': [0.01, 0.001],
    'nu_beta': [0.01, 0.001],
    'nu_alpha': [0.01, 0.001],
    'eta': [0.55, 0.58, 0.6, 0.7],
    'omega': [1e-5, 1e-4, 1e-3, 0.005, 1e-2],  # 0.1效果明显不好
    # 'eta': [0.58, 0.6],
    # 'omega': [1e-2],
    'Th': [360, 280, 240]
    # 'nu_mu': [1e-5],
    # 'nu_beta': [1e-5],
    # 'nu_alpha': [1e-5],
    # 'eta': [0.51, 0.55],
    # 'omega': [0.01],
    # 'Th': [60, 120]
}

# 不变动参数
itr_max = 50
reporting_interval = 1
# err_threshold = 1e-1
err_threshold = 1e-1
K = 2  # 考虑0~K跳邻居的影响
a_mu = 1.001
b_mu = 0.01
a_beta = 1
b_beta = 2


# 参数组合
param_combinations = product(*param_grid.values())
best_G_V_TPR = 0
best_G_X_TPR = 0
best_G_V_FPR = 100
best_G_X_FPR = 100
best_result = {}
for index, params in enumerate(param_combinations):
    current_params = dict(zip(param_grid.keys(), params))

    print("第", index, "次训练参数为：", current_params)

    model = TDCausal(**current_params)

    result = model.learn(result_prefix_path,
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
                         b_mu)

    '''
    保存推理结果
    '''
    with open(result_prefix_path + 'params_' + str(index) + '.pk', 'wb') as f:
        pickle.dump(result, f)

    val_score = evaluate_TDCausal(result_prefix_path + 'params_' + str(index) + '.pk', generate_path_prefix)

    G_V_TPR, G_X_TPR, G_V_FPR, G_X_FPR = val_score[0], val_score[1], val_score[2], val_score[3]

    # and G_V_FPR <= best_G_V_FPR \
    #     and G_X_FPR <= best_G_X_FPR

    # if G_V_TPR >= best_G_V_TPR and G_X_TPR >= best_G_X_TPR:
    if G_V_TPR > 0.7 and G_V_FPR < 0.4:
        best_G_V_TPR = G_V_TPR
        best_G_X_TPR = G_X_TPR
        best_G_V_FPR = G_V_FPR
        best_G_X_FPR = G_X_FPR
        best_result = result

print('Best G_V TPR: %f' % (best_G_V_TPR * 100))
print('Best G_X TPR: %f' % (best_G_X_TPR * 100))
print('Best G_V FPR: %f' % (best_G_V_FPR * 100))
print('Best G_X FPR: %f' % (best_G_X_FPR * 100))
print('Best Params: ', best_result)

'''
保存最佳推理结果
'''
with open(result_prefix_path + 'best_params.pk', 'wb') as f:
    pickle.dump(best_result, f)
''' 
最佳推理结果及参数重新输入评估函数以保存最佳评估结果
'''
evaluate_TDCausal(result_prefix_path + 'best_params.pk', generate_path_prefix)
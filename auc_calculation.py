#!/usr/bin/env python
# coding: utf-8

import pandas as pd
col_name = ['label','pred_prob','rank']
data = pd.read_csv('/Users/pingshitu/Desktop/建模相关代码/AUC计算/auc_calculation_data.csv',
                   header=None,
                   names=col_name,
                   low_memory=False)

# 正样本个数
P = data[data['label'] == 1].shape[0]

# 负样本个数
N = data[data['label'] == 0].shape[0]

# 筛选出正样本
data_positive = data[data['label'] == 1].reset_index(drop=True)

# 对正样本的概率倒序排序，并打上row_number的标签
data_positive['positive_rank'] = pd.DataFrame(list(range(1, data_positive.shape[0]+1)))

# 每个正样本有N个预测概率小于其的负样本 = 正样本在原始数据集（有正有负样本）中对应的row_number - 正样本在新数据集（仅有正样本）中对应的row_number
data_positive['positive_rank_delta'] = data_positive['rank'] - data_positive['positive_rank']

# 计算AUC = sum(每个正样本有N个预测概率小于其的负样本) / 所有正负样本间的组合个数
AUC = data_positive['positive_rank_delta'].sum() / (P*N)
print('AUC = ', AUC)
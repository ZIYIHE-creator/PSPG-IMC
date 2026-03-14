# -*- coding: utf-8 -*-

"""The utilities for ``cp_app``."""
import pandas as pd
import numpy as np
import yaml
from sklearn.utils import resample
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

Kb=1.3806504e-23  # Boltzmann constant in [J/K]
Ph=1.98644586e-23 # Planck constant in [J.cm]
Avo=6.02214076e23 # 1/mol
J2cal=0.2390#焦耳转换为卡路里
th2cm=33.35641#将频率单位太赫兹转换为cm-1

def read_vibspectrum(filename):
    frequencies=[]
    with open(filename) as fi:
        for line in fi.readlines()[3:-1]:
            frequencies.append(float(line.strip().split()[2]))
    return np.array(frequencies)

def read_frequencies_from_mesh(filename):
    mesh=yaml.load(open(filename))
    w=[fr["frequency"] for fr in mesh["phonon"][0]["band"]]
    w=np.array(w)*th2cm
    return w

def cv_from_pdos(temp, pdos):
    pdos=pdos[np.where(pdos[:,0]>0)]
    x = Ph * pdos[:,0] / Kb / temp
    expVal = np.exp(x)
    cv_contributions= np.sum(pdos[:,1:],axis=1)* Avo*Kb * x ** 2 * expVal / (expVal - 1.0) ** 2
    return np.sum(cv_contributions)

def cv_from_dos(temp, totaldos):
    dos=totaldos[np.where(totaldos[:,0]>0)]
    x = Ph * dos[:,0] / Kb / temp
    expVal = np.exp(x)
    cv_contributions= dos[:,1]* Avo*Kb * x ** 2 * expVal / (expVal - 1.0) ** 2
    return np.sum(cv_contributions)

def cv_from_frequencies(temp, freqs):
    freqs=freqs[freqs>0]
    x = Ph * freqs / Kb / temp
    expVal = np.exp(x)
    cv_contributions=Avo*Kb * x ** 2 * expVal / (expVal - 1.0) ** 2
    return np.sum(cv_contributions)

def read_totaldos(filename):
    data=np.loadtxt(filename,skiprows=1)
    data[:,0]*=th2cm
    return data

def read_pdos(filename):
    data=np.loadtxt(filename,skiprows=1)
    data[:,0]*=th2cm
    return data


def add_type_label(mydict,atomtype,name,label):
    if atomtype in mydict:
        mydict[atomtype][name]=label
    else:
        mydict[atomtype]={name:label}
    return mydict

def read_atoms_from_mesh(filename):
    mesh=yaml.load(open(filename))
    w=[fr["frequency"] for fr in mesh["phonon"][0]["band"]]
    w=np.array(w)*th2cm
    return w

def cv_from_pdos_site(temp, pdos,site):
    pdos=pdos[np.where(pdos[:,0]>0)]#只保留第一列大于0的那些行，应该是去掉虚频（代表不稳定的震动模式），但是零频率处也有态密度，但是算上零频率就会导致分母为零
    x = Ph * pdos[:,0] / Kb / temp
    expVal = np.exp(x)
    cv_contributions= pdos[:,site+1]* Avo*Kb * x ** 2 * expVal / (expVal - 1.0) ** 2#相较公式而言多乘了态密度pdos[:,site+1]
    return np.sum(cv_contributions)#不应该是返回单位原子的热容吗？怎么返回总和了？-节点处只有一个原子

def select_structures(nsamples,df):
    selected=set()
    for structure_type in df["structure_type"].unique():
        selected.add(df.loc[df["structure_type"]==structure_type].index.values[0])
        selected.add(df.loc[df["structure_type"]==structure_type].index.values[1])
        if len(selected)>nsamples-1:
            break
    for atom_type in df["atom_types"].unique():
        selected.add(df.loc[df["atom_types"]==atom_type].index.values[0])
        if len(selected)>nsamples-1:
            break
        
    while len(selected)< nsamples:
        selected.add(df.sample(1).index.values[0])
        
    return selected

def select_structures_new(nsamples, df):
    if len(df) < nsamples:
        raise ValueError(f"Requested {nsamples} samples, but DataFrame only has {len(df)} rows.")
    selected_indices = df.sample(n=nsamples, random_state=3982).index
    return selected_indices

def select_structures_shap(nsamples, df, key_features=['Electronegativity_Diff']):
    selected = set()
    for feature in key_features:
        df[f'{feature}_bin'] = pd.qcut(df[feature], q=10, duplicates='drop')
        for bin_val in df[f'{feature}_bin'].unique():
            selected.add(df[df[f'{feature}_bin'] == bin_val].index.values[0])
            if len(selected) > nsamples-1:
                break
        if len(selected) > nsamples-1:
            break
    while len(selected) < nsamples:
        selected.add(df.sample(1,random_state=3982).index.values[0])
    return selected
import pandas as pd
import numpy as np

def select_structures_shap_2(nsamples, df, key_features=['Electronegativity_Diff']):
    selected = set()
    
    # Ensure nsamples is not larger than dataframe size
    nsamples = min(nsamples, len(df))
    
    for feature in key_features:
        if feature not in df.columns:
            continue
        try:
            # Use qcut to bin the feature into 10 quantiles
            df[f'{feature}_bin'] = pd.qcut(df[feature], q=10, duplicates='drop')
            # Get unique bins
            unique_bins = df[f'{feature}_bin'].unique()
            for bin_val in unique_bins:
                # Select first index from each bin
                bin_indices = df[df[f'{feature}_bin'] == bin_val].index
                if len(bin_indices) > 0:
                    selected.add(bin_indices[0])
                if len(selected) >= nsamples:
                    break
            # Drop temporary bin column
            df = df.drop(f'{feature}_bin', axis=1)
        except ValueError as e:
            print(f"Binning failed for {feature}: {e}")
            continue
        if len(selected) >= nsamples:
            break
    
    # Fill remaining slots with random sampling
    if len(selected) < nsamples:
        remaining = nsamples - len(selected)
        # Use random sampling without fixed seed to avoid repetition
        random_indices = df.index.difference(selected)
        if len(random_indices) > 0:
            sampled_indices = np.random.choice(random_indices, size=min(remaining, len(random_indices)), replace=False)
            selected.update(sampled_indices)
    
    return selected

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

def select_structures_active(nsamples, df, models_list, FEATURES, target_col, exclude_indices=None):
    """
    主动学习采样，基于多模型平均残差，支持排除已选样本。

    参数：
    - nsamples: 目标样本数量
    - df: 数据框
    - models_list: Pipeline模型列表
    - key_features: 特征列名（列表）
    - target_col: 目标列名
    - exclude_indices: 可选，已选样本索引（降低重复）

    返回：
    - selected: 包含nsamples个索引的集合
    """
    if not models_list:
        raise ValueError("'models_list' cannot be empty.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in df.")
    if isinstance(FEATURES, str):
        FEATURES = [FEATURES]
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Features not found in df: {missing_features}")

    selected = set()
    X = df[FEATURES]
    y = df[target_col]

    # 计算残差
    all_residuals = []
    for model in models_list:
        y_pred = cross_val_predict(model, X, y, cv=5)
        residuals = np.abs(y - y_pred)
        all_residuals.append(residuals)

    # 平均残差
    uncertainty = np.mean(all_residuals, axis=0)

    # 降低已选样本优先级
    if exclude_indices is not None:
        penalty = 0.5  # 惩罚因子
        uncertainty[[i in exclude_indices for i in df.index]] *= penalty

    # 选择Top-k样本
    df = df.copy()
    df['uncertainty'] = uncertainty
    top_indices = df['uncertainty'].nlargest(nsamples).index
    selected.update(top_indices[:nsamples])

    # 随机补齐
    while len(selected) < nsamples:
        remaining_indices = df.index.difference(selected)
        if len(remaining_indices) > 0:
            selected.add(np.random.choice(remaining_indices))

    df = df.drop('uncertainty', axis=1)
    return selected
def iterative_active_learning(df, models_list, FEATURES, target_col, N_train, max_iterations=10, mse_threshold=0.01):
    """
    迭代主动学习，选择高残差样本直到性能稳定。

    参数：
    - df: 数据框
    - models_list: Pipeline模型列表
    - key_features: 特征列名
    - target_col: 目标列名
    - N_train: 每轮训练样本数
    - max_iterations: 最大迭代次数
    - mse_threshold: MSE变化阈值（停止条件）

    返回：
    - best_model: 最佳模型
    - best_train_indices: 最佳训练集索引
    - history: 性能记录
    """
    history = {'iteration': [], 'train_mse': [], 'test_mse': []}
    best_mse = float('inf')
    best_model = None
    best_train_indices = None
    selected_indices = set()

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")

        # 选择高残差样本
        inds = select_structures_active(
            nsamples=N_train,
            df=df,
            models_list=models_list,
            FEATURES=FEATURES,
            target_col=target_col,
            exclude_indices=selected_indices
        )

        # 更新已选索引（降低重复）
        selected_indices.update(inds)

        # 划分数据集（将set转换为list）
        df_train = df.loc[list(inds)]  # 修复：set -> list
        df_test = df.loc[~df.index.isin(inds)]

        # 训练新模型（使用第一个Pipeline的配置）
        model = Pipeline([
            ('scaling', StandardScaler()),
            ('xgb', XGBRegressor(**models_list[0].named_steps['xgb'].get_params()))
        ])
        model.fit(df_train[FEATURES], df_train[target_col])

        # 评估性能
        y_train_pred = model.predict(df_train[FEATURES])
        y_test_pred = model.predict(df_test[FEATURES])
        train_mse = mean_squared_error(df_train[target_col], y_train_pred)
        test_mse = mean_squared_error(df_test[target_col], y_test_pred)

        # 记录
        history['iteration'].append(iteration + 1)
        history['train_mse'].append(train_mse)
        history['test_mse'].append(test_mse)
        print(f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")

        # 更新最佳模型
        if test_mse < best_mse:
            best_mse = test_mse
            best_model = model
            best_train_indices = inds

        # 检查收敛
        if iteration > 0:
            mse_change = abs(history['test_mse'][-2] - test_mse) / history['test_mse'][-2]
            if mse_change < mse_threshold:
                print(f"Converged: MSE change {mse_change:.4f} < {mse_threshold}")
                break

        # 更新models_list（替换第一个模型）
        models_list = [model] + models_list[1:]

    # 确保best_train_indices是list
    if best_train_indices is not None:
        best_train_indices = list(best_train_indices)

    return best_model, best_train_indices, history


#import numpy as np
#import pandas as pd
#from sklearn.model_selection import cross_val_predict

#def select_structures_active(nsamples, df, models_list, key_features, target_col):
    """
    主动学习采样，适配XGBoost回归Pipeline，基于多模型平均残差选择高不确定性样本。

    参数：
    - nsamples: 目标样本数量
    - df: 数据框，包含特征和目标列
    - models_list: Pipeline对象列表（每个模型包含scaling和xgb步骤）
    - key_features: 使用的特征列表
    - target_col: 目标列名

    返回：
    - selected: 包含nsamples个索引的集合
    """
    # 输入验证
#    if not models_list:
#        raise ValueError("'models_list' cannot be empty.")
#    if key_features is None or target_col is None:
#        raise ValueError("'key_features' and 'target_col' must be provided.")
#    if target_col not in df.columns:
#        raise ValueError(f"Target column '{target_col}' not found in df.")

#    selected = set()
#    X = df[key_features]
#    y = df[target_col]

    # 计算每个模型的残差
#    all_residuals = []
#    for model in models_list:
        # 使用Pipeline的predict方法（自动处理缩放）
#        y_pred = cross_val_predict(model, X, y, cv=5)
#        residuals = np.abs(y - y_pred)  # 绝对残差
#        all_residuals.append(residuals)

    # 平均残差作为不确定性
#    uncertainty = np.mean(all_residuals, axis=0)

    # 选择Top-k样本
#    df = df.copy()
#    df['uncertainty'] = uncertainty
#    top_indices = df['uncertainty'].nlargest(nsamples).index
#    selected.update(top_indices[:nsamples])

    # 随机补齐
#    while len(selected) < nsamples:
#        remaining_indices = df.index.difference(selected)
#        if len(remaining_indices) > 0:
#            selected.add(np.random.choice(remaining_indices))

    # 清理临时列
#    df = df.drop('uncertainty', axis=1)
#    return selected
    
def select_structures_shap_new(nsamples, df, target_col='Value', feature_col='Electronegativity_Diff', bins=10):
    """
    按目标值分箱后，在每个分箱中基于特征值抽样，确保训练集和测试集分布一致性。
    
    Parameters:
    - nsamples: 需要的样本总数
    - df: 输入DataFrame，包含目标值和特征值
    - target_col: 目标值列名（如'Value'）
    - feature_col: 用于抽样的特征列名（如'Electronegativity_Diff'）
    - bins: 目标值分箱数量
    
    Returns:
    - selected: 抽样索引集合
    """
    selected = set()
    
    # 按目标值分箱
    df['target_bin'] = pd.qcut(df[target_col], q=bins, duplicates='drop')
    
    # 计算每个分箱需要的样本数（尽量均匀分配）
    samples_per_bin = max(1, nsamples // len(df['target_bin'].unique()))
    
    # 在每个分箱中基于特征值抽样
    for bin_val in df['target_bin'].unique():
        bin_data = df[df['target_bin'] == bin_val]
        if bin_data.empty:
            continue
            
        # 对特征值分箱以确保多样性
        bin_data['feature_bin'] = pd.qcut(bin_data[feature_col], q=min(5, len(bin_data)), duplicates='drop')
        
        # 在特征分箱中抽样
        for feat_bin in bin_data['feature_bin'].unique():
            feat_bin_data = bin_data[bin_data['feature_bin'] == feat_bin]
            if not feat_bin_data.empty:
                selected.add(feat_bin_data.sample(1).index.values[0])
                if len(selected) >= nsamples:
                    break
        if len(selected) >= nsamples:
            break
    
    # 如果样本不足，随机补充
    while len(selected) < nsamples:
        selected.add(df.sample(1).index.values[0])
    
    # 清理临时列
    df.drop(['target_bin', 'feature_bin'], axis=1, inplace=True, errors='ignore')
    
    return selected
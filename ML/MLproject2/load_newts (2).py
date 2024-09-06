from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_newts(filepath, do_min_max=False):
    data = pd.read_csv(filepath, delimiter=';')
    # print(data.columns)
    xvals_raw = data.drop(['ID', 'Green frogs', 'Brown frogs', 'Common toad', 'Tree frog', 'Common newt', 'Great crested newt', 'Fire-bellied toad'], axis=1)
    xvals = pd.get_dummies(xvals_raw, columns=['Motorway', 'TR', 'VR', 'SUR1', 'SUR2', 'SUR3', 'UR', 'FR', 'MR', 'CR'])
    yvals = data['Fire-bellied toad']

    # optional min-max scaling
    if (do_min_max):
       for col in ['SR', 'NR', 'TR', 'VR', 'OR', 'RR', 'BR']:
           xvals_raw[col] = (xvals_raw[col] - xvals_raw[col].min())/(xvals_raw[col].max() - xvals_raw[col].min())
    xvals = pd.get_dummies(xvals_raw, columns=['Motorway', 'TR', 'VR', 'SUR1', 'SUR2', 'SUR3', 'UR', 'FR', 'MR', 'CR'])
    return xvals, yvals

# 交叉验证情况下，参数C对LinearSVM的影响（F1_cv_c_LSVM）。
# 交叉验证情况下，参数C对KernelSVM的影响(F1_cv_c_KSVM)，参数gamma对Kernel的影响(F1_cv_gamma_KSVM)
def eval_kfold_stub(xvals, yvals,C_max,gamma_max):
    kf = KFold(n_splits = 5, shuffle = True)

    # 记录三种情况的F1值
    F1_cv_c_LSVM = []
    F1_cv_c_KSVM = []
    F1_cv_gamma_KSVM = []

    for train_idxs, test_idxs in kf.split(xvals,yvals):
        xtrain_this_fold = xvals.loc[train_idxs]
        xtest_this_fold = xvals.loc[test_idxs]
        ytrain_this_fold = yvals.loc[train_idxs]
        ytest_this_fold = yvals.loc[test_idxs]
        # train a model on this fold
        f1_this_fold_c_LSVM = []
        for c_params in range(1,C_max):

            # 设置C的范围为0.1-100
            C = round(c_params*0.1,2)
            # print(C)
            linearSVC = LinearSVC(C=C, random_state=10)
            linearSVC.fit(xtrain_this_fold,ytrain_this_fold)
            f1_this_fold_c_LSVM.append(f1_score(ytest_this_fold,linearSVC.predict(xtest_this_fold)))
        F1_cv_c_LSVM.append(f1_this_fold_c_LSVM)   

        f1_this_fold_c_KSVM = []
        for c_params in range(1,C_max):

            # 设置C的范围为0.1-100
            C = round(c_params*0.1,2)
            KernelSVC = SVC(C=C, random_state=10)
            KernelSVC.fit(xtrain_this_fold,ytrain_this_fold)
            f1_this_fold_c_KSVM.append(f1_score(ytest_this_fold,KernelSVC.predict(xtest_this_fold)))
        F1_cv_c_KSVM.append(f1_this_fold_c_KSVM)   

        f1_this_fold_gamma = []
        for gamma_params in range(1,gamma_max):

            # 设置gamma的范围为0.01-10
            gamma = round(gamma_params*0.01,4)
            KernelSVC_1 = SVC(gamma=gamma, random_state=10)
            KernelSVC_1.fit(xtrain_this_fold,ytrain_this_fold)
            f1_this_fold_gamma.append(f1_score(ytest_this_fold,KernelSVC_1.predict(xtest_this_fold)))
        F1_cv_gamma_KSVM.append(f1_this_fold_gamma)  

    return F1_cv_c_LSVM, F1_cv_c_KSVM, F1_cv_gamma_KSVM


# 所有数据训练情况下，参数C对LinearSVM的影响（F1_ad_c_LSVM）。
# 所有数据训练情况下，参数C对KernelSVM的影响(F1_ad_c_KSVM)，参数gamma对Kernel的影响(F1_ad_gamma_KSVM)
def all_dataset(xvals,yvals,C_max,gamma_max):

    # 使用所有数据训练模型，并记录F1值
    F1_ad_c_LSVM = []
    F1_ad_c_KSVM = []
    F1_ad_gamma_KSVM = []

    # 参数C对LinearSVM的影响
    for c_params in range(1,C_max):
        C = round(c_params*0.1,2)
        linearSVC = LinearSVC(C=C,random_state=10)
        linearSVC.fit(xvals,yvals)
        f1 = f1_score(yvals,linearSVC.predict(xvals))
        F1_ad_c_LSVM.append(f1)

    # 参数C对KernelSVM的影响
    for c_params in range(1,C_max):
        C = round(c_params*0.1,2)
        KernelSVC = SVC(C=C,random_state=10)
        KernelSVC.fit(xvals,yvals)
        f1 = f1_score(yvals,KernelSVC.predict(xvals))
        F1_ad_c_KSVM.append(f1)

    # 参数gamma对KernelSVM的影响
    for gamma_params in range(1,gamma_max):
        # 设置gamma的范围为0.01-10
        gamma = round(gamma_params*0.01,4)
        KernelSVC = SVC(gamma=gamma, random_state=10)  
        KernelSVC.fit(xvals,yvals)
        f1_kernelSVM = f1_score(yvals,KernelSVC.predict(xvals))
        F1_ad_gamma_KSVM.append(f1_kernelSVM)

    return F1_ad_c_LSVM, F1_ad_c_KSVM, F1_ad_gamma_KSVM

# 获取最大最小化缩放后的数据
xvals, yvals = load_newts('dataset.csv',do_min_max=True)
print(xvals.shape,yvals.shape)
# print(yvals)

# 获取交叉验证情况下的各个模型的F1值，并且对五折验证取平均.C_max=1001,gamma_max=100000
C_max=1001
gamma_max=1001

F1_cv_c_LSVM, F1_cv_c_KSVM, F1_cv_gamma_KSVM = eval_kfold_stub(xvals,yvals,C_max=C_max,gamma_max=gamma_max)

# 转为numpy数组，方便求均值
F1_cv_c_LSVM = np.array(F1_cv_c_LSVM).mean(axis=0)
F1_cv_c_KSVM = np.array(F1_cv_c_KSVM).mean(axis=0)
F1_cv_gamma_KSVM = np.array(F1_cv_gamma_KSVM).mean(axis=0)


# 获取所有数据训练情况下的F1值
F1_ad_c_LSVM, F1_ad_c_KSVM, F1_ad_gamma_KSVM = all_dataset(xvals,yvals,C_max=C_max,gamma_max=gamma_max)

# 画图
X_axis_C = [round(c_params*0.1,2) for c_params in range(1,C_max)]
X_axis_gamma = [round(gamma_params*0.0001,4) for gamma_params in range(1,gamma_max)]

# 图1
plt.figure()
plt.plot(X_axis_C, F1_cv_c_LSVM,label='cross validation F1 score of LinearSVM')
plt.plot(X_axis_C, F1_ad_c_LSVM,label='all dataset F1 score of LinearSVM')
plt.legend()
plt.xlabel('parameter C')
plt.ylabel('F1 score')
plt.savefig('figure1.jpg')
plt.show()

# 图2
plt.figure()
plt.plot(X_axis_C, F1_cv_c_KSVM,label='cross validation F1 score of KernelSVM')
plt.plot(X_axis_C, F1_ad_c_KSVM,label='all dataset F1 score of KernelSVM')
plt.legend()
plt.xlabel('parameter C')
plt.ylabel('F1 score')
plt.savefig('figure2.jpg')
plt.show()

# 图3
plt.figure()
plt.plot(X_axis_gamma, F1_cv_gamma_KSVM,label='cross validation F1 score of KernelSVM')
plt.plot(X_axis_gamma, F1_ad_gamma_KSVM,label='all dataset F1 score of KernelSVM')
plt.legend()
plt.xlabel('parameter gamma')
plt.ylabel('F1 score')
plt.savefig('figure3.jpg')
plt.show()

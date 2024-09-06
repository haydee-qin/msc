from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np

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

def train_mlp(xvals, yvals):
    metrics = [
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.Accuracy(name="accuracy")
    ]
    F1_mlp = precision_mlp = []
    for hidden_unit in range(1,128):
        model = keras.Sequential(
        [
            keras.layers.Dense(
                56, activation=None, input_shape=(56,)
            ),
            keras.layers.Dense(hidden_unit, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        # print(model.summary())
        model.compile(optimizer=keras.optimizers.SGD(1e-2), loss="binary_crossentropy", metrics=metrics)
        history = model.fit(xvals,yvals,batch_size=32,epochs=100,verbose=1,validation_data=(xvals, yvals))
        val_precision = history.history['val_accuracy']
        val_recall = history.history['val_recall']
        F1_mlp.append((2*val_precision[-1]*val_recall[-1])/(val_precision[-1]+val_recall[-1]))
        precision_mlp.append(val_precision[-1])
        print('nnnnnn',val_precision[-1])

    return F1_mlp,precision_mlp


# 交叉验证MLP
def eval_kfold_stub(xvals, yvals):
    kf = KFold(n_splits = 5, shuffle = True)

    # 记录F1值
    precision_mlp_cv = []
    F1_mlp_cv = []

    for train_idxs, test_idxs in kf.split(xvals,yvals):
        xtrain_this_fold = xvals.loc[train_idxs]
        xtest_this_fold = xvals.loc[test_idxs]
        ytrain_this_fold = yvals.loc[train_idxs]
        ytest_this_fold = yvals.loc[test_idxs]
        # train a model on this fold
        F1_mlp_1,precision_mlp_1 =  train_mlp(xtrain_this_fold, ytrain_this_fold)
        F1_mlp_cv.append(F1_mlp_1)
        precision_mlp_cv.append(precision_mlp_1)

    return F1_mlp_cv,precision_mlp_cv

# MLP model 
xvals, yvals = load_newts('dataset.csv',do_min_max=True)


# 图4
F1_mlp, precision_mlp = train_mlp(xvals, yvals)
X_axis_hidden = [x for x in range(1,5)]

# 交叉验证取平均
F1_mlp_cv,precision_mlp_cv = eval_kfold_stub(xvals, yvals)
print(np.array(F1_mlp_cv).shape)
F1_mlp_cv = np.array(F1_mlp_cv).mean(axis=0)
precision_mlp_cv = np.array(precision_mlp_cv).mean(axis=0)

plt.figure()
plt.plot(X_axis_hidden, precision_mlp[0:len(X_axis_hidden)],label='all dataset precision score of MLP')
plt.plot(X_axis_hidden, precision_mlp_cv[0:len(X_axis_hidden)],label='cross validation precision score of MLP')
plt.legend()
plt.xlabel('the number of hidden units')
plt.ylabel('F1 score')
plt.savefig('figure4.jpg')
plt.show()


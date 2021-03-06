#coding=utf-8
#这个版本对于数据可视化进行了探索并没有效果咯,积累了特征选择等相关功能咯
#实现了lasso和xgboost的超参搜索，并对代码进行了一些优化吧
#他妈了隔壁熬，我之前设置的搜索10000次运行了很久都没运行完熬，改成7000 3000 最后修改成700咯
#然后提交了两个xgboost超参搜索700次的预测结果，第二预测也就是经过log处理的已经进入前50%咯，感觉有进步就是很开心的事情咯
import ast
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectFromModel, RFE, VarianceThreshold

import hyperopt
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

from sklearn.feature_extraction import DictVectorizer

import eli5
from eli5.sklearn import PermutationImportance

import seaborn as sns 
import matplotlib.pyplot as plt
from IPython.display import HTML, display

from catboost import CatBoostRegressor

from xgboost.sklearn import XGBRegressor

#这样的做法主要是为了节约计算时间咯
X_train_scaled = pd.read_csv("train_scaled_5_15.csv", encoding="ANSI")
X_test_scaled = pd.read_csv("test_scaled_5_15.csv", encoding="ANSI")
data_train =  pd.read_csv("train.csv", encoding="ANSI")
data_test = pd.read_csv("test.csv", encoding="ANSI")
Y_train = data_train["revenue"]

def save_inter_params(trials, space_nodes, best_nodes, title):
 
    files = open(str(title+"_intermediate_parameters.pickle"), "wb")
    pickle.dump([trials, space_nodes, best_nodes], files)
    files.close()

def load_inter_params(title):
  
    files = open(str(title+"_intermediate_parameters.pickle"), "rb")
    trials, space_nodes, best_nodes = pickle.load(files)
    files.close()
    
    return trials, space_nodes ,best_nodes

def save_stacked_dataset(stacked_train, stacked_test, title):
    
    files = open(str(title+"_stacked_dataset.pickle"), "wb")
    pickle.dump([stacked_train, stacked_test], files)
    files.close()

def load_stacked_dataset(title):
    
    files = open(str(title+"_stacked_dataset.pickle"), "rb")
    stacked_train, stacked_test = pickle.load(files)
    files.close()
    
    return stacked_train, stacked_test

def save_best_model(best_model, title):
    
    files = open(str(title+"_best_model.pickle"), "wb")
    pickle.dump(best_model, files)
    files.close()

def load_best_model(title_and_nodes):
    
    files = open(str(title_and_nodes+"_best_model.pickle"), "rb")
    best_model = pickle.load(files)
    files.close()
    
    return best_model

#因为sklearn的scoring都是越大越好，所以neg_mean_squared_error会返回负值
#因为lasso_f认为最小的返回值是最佳结果，所以lasso_f需要返回-scoring
def lasso_f(params):
    
    print("alpha", params["alpha"])
    print("fit_intercept", params["fit_intercept"])
    print("normalize", params["normalize"])
    
    rsg = Lasso(alpha = params["alpha"],
                fit_intercept = params["fit_intercept"],
                normalize= params["normalize"],
                random_state = 42)
    
    #skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=None)
    metric = cross_val_score(rsg, X_train_scaled, Y_train, cv=25, scoring="neg_mean_squared_error").mean()
    
    print(-metric)
    #主要我想看rmse的结果，但是metric的结果是负数
    #所以采用下面的形式返回
    print((-metric)**0.5) 
    print()
    return -metric
    
def parse_lasso_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes = {}
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    
    best_nodes["alpha"] = space_nodes["alpha"][trials_list[0]["misc"]["vals"]["alpha"][0]]
    best_nodes["fit_intercept"] = space_nodes["fit_intercept"][trials_list[0]["misc"]["vals"]["fit_intercept"][0]]
    best_nodes["normalize"] = space_nodes["normalize"][trials_list[0]["misc"]["vals"]["normalize"][0]]
    
    return best_nodes

def train_lasso_model(best_nodes, X_train_scaled, Y_train):
    
    rsg = Lasso(alpha = best_nodes["alpha"],
                fit_intercept = best_nodes["fit_intercept"],
                normalize= best_nodes["normalize"],
                random_state = 42)
    rsg.fit(X_train_scaled, Y_train)
    Y_pred = rsg.predict(X_train_scaled)
    print("mse:", np.mean((Y_pred-Y_train)**2))
    print("rmse:", np.sqrt(np.mean((Y_pred-Y_train)**2)))
    return rsg   

lasso_space = {"title":hp.choice("title", ["stacked_tmdb_box_office_prediction"]),
               "path":hp.choice("path", ["TMDB_Box_Office_Prediction.csv"]),
               "mean":hp.choice("mean", [0]),
               "std":hp.choice("std", [0]),
               #这个linspace返回的类型是ndarray类型的数据
               #"feature_num":hp.choice("feature_num", np.linspace(1,385,385)),
               "alpha":hp.choice("alpha", np.logspace(-2, 3, 12)),
               "fit_intercept":hp.choice("fit_intercept", [True, False]),
               "normalize":hp.choice("normalize", [True, False])
               }

lasso_space_nodes = {"title":["stacked_tmdb_box_office_prediction"],
                     "path":["TMDB_Box_Office_Prediction.csv"],
                     "mean":[0],
                     "std":[0],
                     #"feature_num":np.linspace(1,385,385),
                     "alpha":np.logspace(-2, 3, 12),
                     "fit_intercept":[True, False],
                     "normalize":[True, False]
                     }

#因为sklearn的scoring都是越大越好，所以neg_mean_squared_error会返回负值
#因为xgb_f认为最小的返回值是最佳结果，所以xgb_f需要返回-scoring
def xgb_f(params):
    
    print("title", params["title"])
    print("path", params["path"])
    print("mean", params["mean"])
    print("std", params["std"])
    #print("feature_num", params["feature_num"])
    print("gamma", params["gamma"]) 
    print("max_depth", params["max_depth"])
    print("learning_rate", params["learning_rate"])
    print("min_child_weight", params["min_child_weight"])
    print("subsample", params["subsample"])
    print("colsample_bytree", params["colsample_bytree"])
    print("reg_alpha", params["reg_alpha"])
    print("reg_lambda", params["reg_lambda"])
    print("n_estimators", params["n_estimators"])

    clf = XGBRegressor(gamma=params["gamma"],
                       max_depth=params["max_depth"],
                       learning_rate=params["learning_rate"],
                       min_child_weight=params["min_child_weight"],
                       subsample=params["subsample"],
                       colsample_bytree=params["colsample_bytree"],
                       reg_alpha=params["reg_alpha"],
                       reg_lambda=params["reg_lambda"],
                       n_estimators=int(params["n_estimators"]),
                       random_state=42)

    skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(clf, X_train_scaled, Y_train, cv=skf, scoring="neg_mean_squared_error").mean()
    
    print(-metric)
    #主要我想看rmse的结果，但是metric的结果是负数
    #所以采用下面的形式返回
    print((-metric)**0.5) 
    print()
    return -metric
    
def parse_xgb_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes = {}
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    
    #best_nodes["feature_num"] = space_nodes["feature_num"][trials_list[0]["misc"]["vals"]["feature_num"][0]]
    best_nodes["gamma"] = space_nodes["gamma"][trials_list[0]["misc"]["vals"]["gamma"][0]]
    best_nodes["max_depth"] = space_nodes["max_depth"][trials_list[0]["misc"]["vals"]["max_depth"][0]]
    best_nodes["learning_rate"] = space_nodes["learning_rate"][trials_list[0]["misc"]["vals"]["learning_rate"][0]]
    best_nodes["min_child_weight"] = space_nodes["min_child_weight"][trials_list[0]["misc"]["vals"]["min_child_weight"][0]]
    best_nodes["subsample"] = space_nodes["subsample"][trials_list[0]["misc"]["vals"]["subsample"][0]]
    best_nodes["colsample_bytree"] = space_nodes["colsample_bytree"][trials_list[0]["misc"]["vals"]["colsample_bytree"][0]]
    best_nodes["reg_alpha"] = space_nodes["reg_alpha"][trials_list[0]["misc"]["vals"]["reg_alpha"][0]]
    best_nodes["reg_lambda"] = space_nodes["reg_lambda"][trials_list[0]["misc"]["vals"]["reg_lambda"][0]]
    best_nodes["n_estimators"] = space_nodes["n_estimators"][trials_list[0]["misc"]["vals"]["n_estimators"][0]]

    return best_nodes

def train_xgb_model(best_nodes, X_train_scaled, Y_train):
    
    clf = XGBRegressor(gamma=best_nodes["gamma"],
                       max_depth=best_nodes["max_depth"],
                       learning_rate=best_nodes["learning_rate"],
                       min_child_weight=best_nodes["min_child_weight"],
                       subsample=best_nodes["subsample"],
                       colsample_bytree=best_nodes["colsample_bytree"],
                       reg_alpha=best_nodes["reg_alpha"],
                       reg_lambda=best_nodes["reg_lambda"],
                       n_estimators=int(best_nodes["n_estimators"]),
                       random_state=42)

    rsg.fit(X_train_scaled, Y_train)
    Y_pred = rsg.predict(X_train_scaled)
    print("mse:", np.mean((Y_pred-Y_train)**2))
    print("rmse:", np.sqrt(np.mean((Y_pred-Y_train)**2)))
    return rsg   

xgb_space = {"title":hp.choice("title", ["stacked_don't_overfit!_II"]),
             "path":hp.choice("path", ["Don't_Overfit!_II_Prediction.csv"]),
             "mean":hp.choice("mean", [0]),
             "std":hp.choice("std", [0]),
             #"feature_num":hp.choice("feature_num", np.linspace(1,300,300)),
             "gamma":hp.choice("gamma", [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.3, 0.5, 0.7, 0.9]),
             "max_depth":hp.choice("max_depth", [3, 5, 7, 9, 12, 15, 17, 25]),
             "learning_rate":hp.choice("learning_rate", np.linspace(0.01, 0.50, 50)),
             "min_child_weight":hp.choice("min_child_weight", [1, 3, 5, 7, 9]),
             "subsample":hp.choice("subsample", [0.6, 0.7, 0.8, 0.9, 1.0]),
             "colsample_bytree":hp.choice("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1.0]),
             "reg_alpha":hp.choice("reg_alpha", [0.0, 0.1, 0.5, 1.0]),
             "reg_lambda":hp.choice("reg_lambda", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 1.0]),
             "n_estimators":hp.choice("n_estimators", np.linspace(50, 500, 46))
             }

xgb_space_nodes = {"title":["stacked_don't_overfit!_II"],
                   "path":["Don't_Overfit!_II_Prediction.csv"],
                   "mean":[0],
                   "std":[0],
                   #"feature_num":np.linspace(1,300,300),
                   "gamma":[0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.3, 0.5, 0.7, 0.9],
                   "max_depth":[3, 5, 7, 9, 12, 15, 17, 25],
                   "learning_rate":np.linspace(0.01, 0.50, 50),
                   "min_child_weight":[1, 3, 5, 7, 9],
                   "subsample":[0.6, 0.7, 0.8, 0.9, 1.0],
                   "colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
                   "reg_alpha":[0.0, 0.1, 0.5, 1.0],
                   "reg_lambda":[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 1.0],
                   "n_estimators":np.linspace(50, 500, 46)
                   }

"""
rfc_model = Lasso(random_state=42).fit(X_train_scaled, Y_train)
perm = PermutationImportance(rfc_model, random_state=42).fit(X_train_scaled, Y_train)
eli5.show_weights(perm, feature_names = X_train_scaled.columns.tolist())
"""

"""
rfc_model = Lasso(random_state = 42).fit(X_train_scaled, Y_train)
perm = PermutationImportance(rfc_model, random_state=42).fit(X_train_scaled, Y_train)
feature_importances1 = perm.feature_importances_#这是返回每个特征的权重
feature_importances_std = perm.feature_importances_std_ 
feature_importances2 = np.where(feature_importances1>0)#此时我记录下了每个特征的列数
X_train_scaled_new = X_train_scaled[X_train_scaled.columns[feature_importances2]]
X_test_scaled_new = X_test_scaled[X_test_scaled.columns[feature_importances2]]
"""

"""
rfc_model = XGBRegressor(random_state = 42).fit(X_train_scaled, Y_train) #选了62个特征，怎么相差这么多呢
#rfc_model = Lasso(random_state = 42).fit(X_train_scaled, Y_train) #选了269个特征，我了个去
perm = PermutationImportance(rfc_model, random_state=42).fit(X_train_scaled, Y_train)
feature_importances1 = perm.feature_importances_#这是返回每个特征的权重
feature_importances_std = perm.feature_importances_std_ 
feature_importances2 = np.where(feature_importances1>0)#此时我记录下了每个特征的列数
X_train_scaled_new = X_train_scaled[X_train_scaled.columns[feature_importances2]]
X_test_scaled_new = X_test_scaled[X_test_scaled.columns[feature_importances2]]
"""

#然后看一哈数据的分布，如果数据不符合正态分布的话，还需要进一步进行调整
"""
#下面的实验表明，取log的效果总体而言比不取更符合正态分布
#至于为什么要取正态分布，因为正态分布的数据更适合拟合
#至于取np.log1p还是np.log感觉基本差不多的
sns.distplot(data_train["revenue"], rug=True)
plt.show()
#plt.hist(data_train["revenue"], bins=50, color='steelblue', normed=True )
#temp = np.log1p(data_train["revenue"])
temp1 = np.log(data_train["revenue"])
sns.distplot(temp1, rug=True)
plt.show()
temp2 = np.log1p(data_train["revenue"])
sns.distplot(temp2, rug=True)
plt.show()#我了个透不写这个确实是不会显示信息的
temp3 = np.log10(data_train["revenue"])
sns.distplot(temp3, rug=True)
plt.show()
temp4 = np.log2(data_train["revenue"])
sns.distplot(temp4, rug=True)
plt.show()
np.logaddexp
"""
#太奇怪了，用了下面这个log之后，lasso模型的PermutationImportance就选不出特征了
#但是XGBRegressor原来只能够选出62个特征的，现在选出了69个特征。。
#那就都试一哈吧，看看到底是什么情况熬，我觉得数据科学真的是玄学熬。。
#Y_train = np.log(Y_train)

"""
#说实话我觉得这些数据看了可能都妹啥用，特征选择就交给算法吧？
data_train[['revenue', 'budget']].groupby(['budget']).mean().plot.bar()
plt.show()
data_train[['revenue', 'runtime']].groupby(['runtime']).mean().plot.bar()
"""

"""
#这样吧，我今天先用lasso xgboost 以及catboost分别提交一个超参搜索的版本吧
#lasso的版本无法进行Y_train = np.log(Y_train)否则无法选择出特征咯
rfc_model = Lasso(random_state=42).fit(X_train_scaled, Y_train)
#rfc_model = XGBRegressor(random_state=42).fit(X_train_scaled, Y_train)
perm = PermutationImportance(rfc_model, random_state=42).fit(X_train_scaled, Y_train)
feature_importances1 = perm.feature_importances_#这是返回每个特征的权重
feature_importances_std = perm.feature_importances_std_ 
feature_importances2 = np.where(feature_importances1>0)#此时我记录下了每个特征的列数
X_train_scaled_new = X_train_scaled[X_train_scaled.columns[feature_importances2]]
X_test_scaled_new = X_test_scaled[X_test_scaled.columns[feature_importances2]]
X_train_scaled = X_train_scaled_new
X_test_scaled = X_test_scaled_new

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best = fmin(lasso_f, lasso_space, algo=tpe.suggest, max_evals=2, trials=trials)
best_nodes = parse_lasso_nodes(trials, lasso_space_nodes)
save_inter_params(trials, lasso_space_nodes, best_nodes, "tmdb_box_office_prediction")
rsg = train_lasso_model(best_nodes, X_train_scaled, Y_train)

Y_pred = rsg.predict(X_test_scaled)
data = {"id":data_test["id"], "revenue":Y_pred}
output = pd.DataFrame(data = data)            
output.to_csv("lasso_predicton.csv", index=False)
"""

rfc_model = XGBRegressor(random_state=42).fit(X_train_scaled, Y_train)
perm = PermutationImportance(rfc_model, random_state=42).fit(X_train_scaled, Y_train)
feature_importances1 = perm.feature_importances_#这是返回每个特征的权重
feature_importances_std = perm.feature_importances_std_ 
feature_importances2 = np.where(feature_importances1>0)#此时我记录下了每个特征的列数
X_train_scaled_new = X_train_scaled[X_train_scaled.columns[feature_importances2]]
X_test_scaled_new = X_test_scaled[X_test_scaled.columns[feature_importances2]]
X_train_scaled = X_train_scaled_new
X_test_scaled = X_test_scaled_new

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best = fmin(xgb_f, xgb_space, algo=tpe.suggest, max_evals=1, trials=trials)#一共这么多种组合1012000000
best_nodes = parse_xgb_nodes(trials, xgb_space_nodes)
save_inter_params(trials, xgb_space_nodes, best_nodes, "tmdb_box_office_prediction")
rsg = train_xgb_model(best_nodes, X_train_scaled, Y_train)

Y_pred = rsg.predict(X_test_scaled)
data = {"id":data_test["id"], "revenue":Y_pred}
output = pd.DataFrame(data = data)            
output.to_csv("xgb_predicton_1.csv", index=False)


#这个是将Y_train进行log之后的版本，不知道是否能够起到作用呢，我个人推测应该不行吧
Y_train = np.log(Y_train)
rfc_model = XGBRegressor(random_state=42).fit(X_train_scaled, Y_train)
perm = PermutationImportance(rfc_model, random_state=42).fit(X_train_scaled, Y_train)
feature_importances1 = perm.feature_importances_#这是返回每个特征的权重
feature_importances_std = perm.feature_importances_std_ 
feature_importances2 = np.where(feature_importances1>0)#此时我记录下了每个特征的列数
X_train_scaled_new = X_train_scaled[X_train_scaled.columns[feature_importances2]]
X_test_scaled_new = X_test_scaled[X_test_scaled.columns[feature_importances2]]
X_train_scaled = X_train_scaled_new
X_test_scaled = X_test_scaled_new

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best = fmin(xgb_f, xgb_space, algo=tpe.suggest, max_evals=1, trials=trials)#一共这么多种组合1012000000
best_nodes = parse_xgb_nodes(trials, xgb_space_nodes)
save_inter_params(trials, xgb_space_nodes, best_nodes, "tmdb_box_office_prediction")
rsg = train_xgb_model(best_nodes, X_train_scaled, Y_train)
T_pred = rsg.predict(X_train_scaled)
T_pred = np.exp(T_pred)
Y_train = np.exp(Y_train)
print("mse:", np.mean((T_pred-Y_train)**2))
print("rmse:", np.sqrt(np.mean((T_pred-Y_train)**2)))

Y_pred = rsg.predict(X_test_scaled)
Y_pred = np.exp(Y_pred)
data = {"id":data_test["id"], "revenue":Y_pred}
output = pd.DataFrame(data = data)            
output.to_csv("xgb_predicton_2.csv", index=False)
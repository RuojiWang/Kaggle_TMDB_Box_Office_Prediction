#coding=utf-8
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
    metric = cross_val_score(rsg, X_train_scaled, Y_train, cv=10, scoring="neg_mean_squared_error").mean()
    
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

    rsg = XGBRegressor(gamma=params["gamma"],
                       max_depth=params["max_depth"],
                       learning_rate=params["learning_rate"],
                       min_child_weight=params["min_child_weight"],
                       subsample=params["subsample"],
                       colsample_bytree=params["colsample_bytree"],
                       reg_alpha=params["reg_alpha"],
                       reg_lambda=params["reg_lambda"],
                       n_estimators=int(params["n_estimators"]),
                       random_state=42)

    #skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(rsg, X_train_scaled, Y_train, cv=10, scoring="neg_mean_squared_error").mean()
    
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
    
    rsg = XGBRegressor(gamma=best_nodes["gamma"],
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
             "learning_rate":hp.choice("learning_rate", np.linspace(0.01, 0.30, 30)),
             "min_child_weight":hp.choice("min_child_weight", [1, 3, 5, 7, 9]),
             "subsample":hp.choice("subsample", [0.6, 0.7, 0.8, 0.9, 1.0]),
             "colsample_bytree":hp.choice("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1.0]),
             "reg_alpha":hp.choice("reg_alpha", [0.0, 0.1, 0.5, 1.0]),
             "reg_lambda":hp.choice("reg_lambda", [0.01, 0.03, 0.05, 0.07, 0.09, 0.5, 0.7, 0.9, 1.0]),
             "n_estimators":hp.choice("n_estimators", np.linspace(50, 300, 26))
             }

xgb_space_nodes = {"title":["stacked_don't_overfit!_II"],
                   "path":["Don't_Overfit!_II_Prediction.csv"],
                   "mean":[0],
                   "std":[0],
                   #"feature_num":np.linspace(1,300,300),
                   "gamma":[0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.3, 0.5, 0.7, 0.9],
                   "max_depth":[3, 5, 7, 9, 12, 15, 17, 25],
                   "learning_rate":np.linspace(0.01, 0.30, 30),
                   "min_child_weight":[1, 3, 5, 7, 9],
                   "subsample":[0.6, 0.7, 0.8, 0.9, 1.0],
                   "colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
                   "reg_alpha":[0.0, 0.1, 0.5, 1.0],
                   "reg_lambda":[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 1.0],
                   "n_estimators":np.linspace(50, 300, 26)
                   }

cat_space = {"title":hp.choice("title", ["stacked_don't_overfit!_II"]),
             "path":hp.choice("path", ["Don't_Overfit!_II_Prediction.csv"]),
             "mean":hp.choice("mean", [0]),
             "std":hp.choice("std", [0]),
             "iterations":hp.choice("iterations", np.linspace(500, 800, 4)),
             "learning_rate":hp.choice("learning_rate", np.linspace(0.01, 0.30, 30)),
             "depth":hp.choice("depth", [3, 4, 5, 6, 8, 9, 11]),
             #"l2_leaf_reg":hp.choice("l2_leaf_reg", [3, 4, 5, 6, 8, 9, 11]),
             #"n_estimators":hp.choice("n_estimators", [3, 4, 5, 6, 8, 9, 11]),
             #"loss_function":hp.choice("loss_function", ["RMSE"]),
             "custom_metric":hp.choice("custom_metric", ["RMSE"]),
             "partition_random_seed":hp.choice("partition_random_seed", [42]),
             "n_estimators":hp.choice("n_estimators", np.linspace(50, 500, 46))
             }

cat_space_nodes = {"title":["stacked_don't_overfit!_II"],
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

#mother fucker终于找到catboost的调参指南了 https://www.jqr.com/article/000136
#https://blog.csdn.net/linxid/article/details/80723811
#https://blog.csdn.net/AiirrrrYee/article/details/78224232
#我仔细看了一下我觉得catboost还是挺值得学习的一个模型，因为综合性能不输xgboost
rsg = CatBoostRegressor()


"""
#这样吧，我今天先用lasso xgboost 以及catboost分别提交一个超参搜索的版本吧
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
output.to_csv("xgb_predicton.csv", index=False)
"""

"""
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
output.to_csv("xgb_predicton.csv", index=False)
"""

#首先是总结一下house-prices的相关的kernel
#今天看了一下https://www.kaggle.com/tripidhoble/house-prices-predictions
#我觉得别人在这个缺失数据方面做得非常的细致，在填充之前先确认了缺失的程度（缺失百分比）
#卧槽还可以直接使用from sklearn.impute import SimpleImputer填充缺失的数字数据和种类数据
#但是为什么他们每次比赛的时候都需要做一次相关性检验呀，难道直接给学习器让它自行选择不好吗
#可以在一开始的时候直接调用train_data.isnull().sum()
#有些重要的数据可以通过分类器对重要的缺失参数进行赋值，比较典型的模型就是逻辑回归之类的模型
#这个house_prices根本没有做一些特征选择或者特征处理的相关东西。
#https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
#上面的kernel In[8]的部分好像明显有问题的吧，应该用类似one-hot编码的方式进行实现
#In[9]中的create feature simplifications of existing features似乎有待商榷呀
#In[9]中的create feature 2* Combinations of existing features似乎有点东西 加深了我对特征工程的理解
#In[10]中的 Find most important features relative to target corr = train.corr()
#In[11]中还创造了三次的特征关系，我觉得以后这些东西可能需要用库自动排列组合特征了吧？
#In[12]中categorical_features和numerical_features感觉做的蛮好的
#In[14]中skewness计算并对超过0.5的数据进行log
#进行了四个模型的训练，然后才对特征进行了选择（使用和丢弃）
#https://www.kaggle.com/humananalog/xgboost-lasso
#这个kernel使用了from sklearn.metrics import mean_squared_error
#这里的特征工程做的特别的细，感觉很难理解其中的含义吧
#特征缩放之前先进行skewed处理特征咯
#https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
#这里面的Outliers可能以后需要借鉴的吧
#In[7]之类的画图可能也是我们需要积累的吧
#boxcox1p也是log以外的一种数据处理方式咯
#我真的觉得这些狗币看了数据的相关性或者是分布之后并没有做什么卵事情呢

#然后是总结一下titanic的相关的kernel
#https://www.kaggle.com/startupsci/titanic-data-science-solutions
#基本上面就是完整的数据科学处理流程，不论是使用神经网络还是采用传统模型都可以借鉴吧
#剩下的部分都在我的代码里面咯，所以神经网络就是很花计算资源比较吃数据总量，不用做特征工程很方便

#现在总结一下通用的机器学习流程吧
#（1）在填充之前先确认了缺失的程度（缺失百分比）
#（2）然后使用均值和众数填充缺失值
#（3）使用skewness，并查看数据的分布或者与待拟合数据关系
#（4）然后对数据进行相关性检查，并对重要缺失数据采用简单模型进行拟合
#（5）然后是各种尝试从现有的特征中创造出新的特征加入到模型中咯
#（6）在输入到学习器之前先进行一次特征选择熬
#（7）我觉得神经网络的部分就是自己增加噪音咯，以获取更多的数据咯

#然后是阅读这个比赛的kernel
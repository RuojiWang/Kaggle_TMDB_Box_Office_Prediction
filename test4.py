#coding=utf-8
#这个版本总结了一下titanic、house-prices的相关的kernel
#现在总结一下通用的机器学习流程吧
#（1）在填充之前先确认了缺失的程度（缺失百分比）
#（2）然后使用均值和众数填充缺失值
#（3）使用skewness，并查看数据的分布或者与待拟合数据关系
#（4）然后对数据进行相关性检查，并对重要缺失数据采用简单模型进行拟合
#（5）然后是各种尝试从现有的特征中创造出新的特征加入到模型中咯
#（6）在输入到学习器之前先进行一次特征选择熬，特征选择和组合不同的模型相差很多（还有使用肉眼选择的），不再统一归纳。
#（7）我觉得神经网络的部分就是自己增加噪音咯，以获取更多的数据咯

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

#我试了一下这个cat模型的超参搜索，效果好像真的是比xgb强一个层次诶。。我花了很多时间读carboost的参数
#终于找到catboost的调参指南了 https://www.jqr.com/article/000136
#https://blog.csdn.net/linxid/article/details/80723811
#https://blog.csdn.net/AiirrrrYee/article/details/78224232
#先实现这个catboost模型的超参搜索咯，我预计应该比xgboost结果好10%左右吧

#修改出了这个问题的神经网络的模型版本代码

#实现了lightgbm版本的代码咯
#经过我的了解，其实Lightgbm还是有好处的，毕竟xgb训练的速度太慢了
#这个lightgbm的训练速度比起xgb有明显的提升而且准确率等并没有啥损失
#之后的超参搜索主要根据下面的链接中的内容
#https://zhuanlan.zhihu.com/p/27916208
#https://www.cnblogs.com/bjwu/p/9307344.html
#https://juejin.im/post/5b76437ae51d45666b5d9b05

#对比了一下xgb cat lgb进行特征选择之后的模型效果如何

#cat iterations的选择在测试集上面的效果如何

#cat模型和xgb模型进行的一次小规模的battle咯
import ast
import math
import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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

import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F

import skorch
from skorch import NeuralNetRegressor
from lightgbm.sklearn import LGBMRegressor

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

#lasso相关的部分
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

#xgb相关的部分
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

#cat相关的部分
def cat_f(params):
    
    print("title", params["title"])
    print("path", params["path"])
    print("mean", params["mean"])
    print("std", params["std"])
    print("iterations", params["iterations"])
    print("learning_rate", params["learning_rate"])
    print("depth", params["depth"])
    print("l2_leaf_reg", params["l2_leaf_reg"])
    print("custom_metric", params["custom_metric"])

    rsg = CatBoostRegressor(iterations=params["iterations"],
                            learning_rate=params["learning_rate"],
                            depth=params["depth"],
                            l2_leaf_reg=params["l2_leaf_reg"],
                            custom_metric=params["custom_metric"],
                            task_type='GPU')

    #skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(rsg, X_train_scaled, Y_train, cv=10, scoring="neg_mean_squared_error").mean()
    
    print(-metric)
    #主要我想看rmse的结果，但是metric的结果是负数
    #所以采用下面的形式返回
    print((-metric)**0.5) 
    print()
    return -metric
    
def parse_cat_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes = {}
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    
    best_nodes["iterations"] = space_nodes["iterations"][trials_list[0]["misc"]["vals"]["iterations"][0]]
    best_nodes["learning_rate"] = space_nodes["learning_rate"][trials_list[0]["misc"]["vals"]["learning_rate"][0]]
    best_nodes["depth"] = space_nodes["depth"][trials_list[0]["misc"]["vals"]["depth"][0]]
    best_nodes["l2_leaf_reg"]= space_nodes["l2_leaf_reg"][trials_list[0]["misc"]["vals"]["l2_leaf_reg"][0]]
    best_nodes["custom_metric"] = space_nodes["custom_metric"][trials_list[0]["misc"]["vals"]["custom_metric"][0]]
    
    return best_nodes

def train_cat_model(best_nodes, X_train_scaled, Y_train):
    
    rsg = CatBoostRegressor(iterations=best_nodes["iterations"],
                            learning_rate=best_nodes["learning_rate"],
                            depth=best_nodes["depth"],
                            l2_leaf_reg=best_nodes["l2_leaf_reg"],
                            custom_metric=best_nodes["custom_metric"],
                            task_type='GPU')

    rsg.fit(X_train_scaled, Y_train)
    Y_pred = rsg.predict(X_train_scaled)
    print("mse:", np.mean((Y_pred-Y_train)**2))
    print("rmse:", np.sqrt(np.mean((Y_pred-Y_train)**2)))
    return rsg

#其他的参数暂时不知道如何设置，暂时就用这些超参吧，我觉得主要还是特征的创建咯
cat_space = {"title":hp.choice("title", ["stacked_tmdb_box_office_prediction"]),
             "path":hp.choice("path", ["TMDB_Box_Office_Prediction.csv"]),
             "mean":hp.choice("mean", [0]),
             "std":hp.choice("std", [0]),
             "iterations":hp.choice("iterations", [800, 1000, 1200, 1400]),
             "learning_rate":hp.choice("learning_rate", np.linspace(0.01, 0.30, 30)),
             "depth":hp.choice("depth", [3, 4, 5, 6, 8, 9, 11]),
             "l2_leaf_reg":hp.choice("l2_leaf_reg", [2, 3, 5, 7, 9]),
             #"n_estimators":hp.choice("n_estimators", [3, 4, 5, 6, 8, 9, 11]),
             #"loss_function":hp.choice("loss_function", ["RMSE"]),#这个就是默认的
             "custom_metric":hp.choice("custom_metric", ["RMSE"]),
             #"partition_random_seed":hp.choice("partition_random_seed", [42]),#这个是cv里面才用的参数
             #"n_estimators":hp.choice("n_estimators", np.linspace(50, 500, 46)),
             #"border_count":hp.choice("border_count", [16, 32, 48, 64, 96, 128]),
             #"ctr_border_count":hp.choice("ctr_border_count", [32, 48, 50, 64, 96, 128])
             }

cat_space_nodes = {"title":["stacked_tmdb_box_office_prediction"],
                   "path":["TMDB_Box_Office_Prediction.csv"],
                   "mean":[0],
                   "std":[0],
                   #"feature_num":np.linspace(1,300,300),
                   "iterations":[800, 1000, 1200, 1400],
                   "learning_rate":np.linspace(0.01, 0.30, 30),
                   "depth":[3, 4, 5, 6, 8, 9, 11],
                   "l2_leaf_reg":[2, 3, 5, 7, 9],
                   #"n_estimators":[3, 4, 5, 6, 8, 9, 11],
                   #"loss_function":["RMSE"],#这个就是默认的
                   "custom_metric":["RMSE"],
                   #"partition_random_seed":[42],
                   #"n_estimators":np.linspace(50, 500, 46),
                   #"border_count":[16, 32, 48, 64, 96, 128],
                   #"ctr_border_count":[32, 48, 50, 64, 96, 128]
                   }

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        
    def forward(self, pred, truth):
        return torch.sqrt(torch.mean((pred-truth)**2))

def create_nn_module(input_nodes, hidden_layers, hidden_nodes, output_nodes, percentage=0.1):
    
    module_list = []
    
    #当没有隐藏节点的时候
    if(hidden_layers==0):
        module_list.append(nn.Linear(input_nodes, output_nodes))
        module_list.append(nn.Dropout(percentage))
        module_list.append(nn.ReLU())
        #这边softmax的值域刚好就是(0,1)算是符合softmax的值域吧。
        module_list.append(nn.Linear(hidden_nodes, output_nodes))
        
    #当存在隐藏节点的时候
    else :
        module_list.append(nn.Linear(input_nodes, hidden_nodes))
        module_list.append(nn.Dropout(percentage))
        module_list.append(nn.ReLU())
        
        for i in range(0, hidden_layers):
            module_list.append(nn.Linear(hidden_nodes, hidden_nodes))
            module_list.append(nn.Dropout(percentage))
            module_list.append(nn.ReLU())
             
        module_list.append(nn.Linear(hidden_nodes, output_nodes))
            
    model = nn.Sequential()
    for i in range(0, len(module_list)):
        model.add_module(str(i+1), module_list[i])
    
    return model

def init_module(rsg, weight_mode, bias):

    for layer in rsg.modules():
        #我本来想单独处理nn.Linear nn.Conv2d等
        #但是处理的流程好像都是一样的，所以不用分开处理
        #不对啊，nn.Conv2d
        #目前这个可能会遇到dropout relu之类的
        if isinstance(layer, nn.Linear):
            if (weight_mode==1):
                pass
                #weight和bias均使用pytorch默认的初始值，也就是下面的赋值方式咯
                #def reset_parameters(self):
                #stdv = 1. / math.sqrt(self.weight.size(1))
                #self.weight.data.uniform_(-stdv, stdv)
                #if self.bias is not None:
                #    self.bias.data.uniform_(-stdv, stdv)

            #这下面是xavier_normal_的方式初始化的三种情况
            elif (weight_mode==2):
                #使用xavier_normal_的方式初始化weight但是不改变默认bias
                #可能两个函数的std存在差异，不知道是否能够交融在一起
                #现在需要注意的是每个函数需要注意参数，使用默认参数可能造成weight和bias范围不搭
                #uniform_的默认stdv值是stdv = 1. / math.sqrt(self.weight.size(1))
                #xavier_normal_的默认std值是std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                #还好这两个参数的范围是搭的所以可以直接放在一起吧
                nn.init.xavier_normal_(layer.weight.data)
                
            elif (weight_mode==3):
                #使用xavier_normal_的方式初始化weight和bias
                nn.init.xavier_normal_(layer.weight.data)
                #nn.init.xavier_normal_(layer.bias.data)
                #这下面是模拟上面xavier_normal_的初始化方式
                #特别注意了一下bias的std范围和xavier_normal_的是搭配的
                std = 1.0 * math.sqrt(1.0 / layer.weight.data.size(0))
                #这里不计算std的话，可能参数的范围和上面是不搭的
                nn.init.normal_(layer.bias.data, 0, std)
                
            elif (weight_mode==4):
                #使用xavier_normal_的方式初始化weight
                #将bias的值设置为固定的值咯，这样一来好像需要设定很多bias候选值？
                #感觉就提供五到十个选择吧，我理解bias区间范围还是比较窄的（从默认赋值看出）
                nn.init.xavier_normal_(layer.weight.data)
                nn.init.constant_(layer.bias.data, bias)
                
            #接下来是xavier_uniform_初始化的三类情况
            elif (weight_mode==5):
                #xavier_uniform_的std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                #std = math.sqrt(3.0) * std
                #这个std的默认范围和默认的bias的空间是不搭的，但是还是作为一种初始化方案吧
                nn.init.xavier_uniform_(layer.weight.data)
            
            elif (weight_mode==6):
                nn.init.xavier_uniform_(layer.weight.data)
                #nn.init.xavier_uniform_(layer.bias.data)
                #这下面是模拟上面xavier_uniform_的初始化方式
                std = 1.0 * math.sqrt(1.0 / layer.weight.data.size(0))
                std = math.sqrt(3.0) * std
                #这里不计算std的话，可能参数的范围和上面是不搭的
                nn.init.uniform_(layer.bias.data, -std, std)
                
            elif (weight_mode==7):
                nn.init.xavier_uniform_(layer.weight.data)
                nn.init.constant_(layer.bias.data, bias)
            
            #接下来是kaiming_normal_初始化的三类情况
            elif (weight_mode==8):
                #使用kaiming_normal_的方式初始化weight但是不改变默认bias
                nn.init.kaiming_normal_(layer.weight.data)
                
            elif (weight_mode==9):
                nn.init.kaiming_normal_(layer.weight.data)
                gain = math.sqrt(2.0 / (1 + 0** 2))
                fan = layer.bias.data.size(0)
                std = gain / fan
                nn.init.normal_(layer.bias.data, 0, std)
            
            elif (weight_mode==10):
                nn.init.kaiming_normal_(layer.weight.data)
                nn.init.constant_(layer.bias.data, bias)
                
            #接下来是kaiming_uniform_初始化的三类情况
            elif (weight_mode==11):
                #使用kaiming_uniform_的方式初始化weight但是不改变默认bias
                nn.init.kaiming_uniform_(layer.weight.data)
                
            elif (weight_mode==12):
                nn.init.kaiming_uniform_(layer.weight.data)
                gain = math.sqrt(2.0 / (1 + 0** 2))
                fan = layer.bias.data.size(0)
                std = gain / math.sqrt(fan)
                std = math.sqrt(3.0) * std
                nn.init.uniform_(layer.bias.data, -std, std)
            
            elif (weight_mode==13):
                nn.init.kaiming_uniform_(layer.weight.data)
                nn.init.constant_(layer.bias.data, bias)                
                
            else:
                pass

def nn_f(params):
    
    print("mean", params["mean"])
    print("std", params["std"])
    print("lr", params["lr"])
    print("optimizer__weight_decay", params["optimizer__weight_decay"])
    print("criterion", params["criterion"])
    print("batch_size", params["batch_size"])
    print("optimizer__betas", params["optimizer__betas"])
    print("bias", params["bias"])
    print("weight_mode", params["weight_mode"])
    print("patience", params["patience"])
    print("input_nodes", params["input_nodes"])
    print("hidden_layers", params["hidden_layers"])
    print("hidden_nodes", params["hidden_nodes"])
    print("output_nodes", params["output_nodes"])
    print("percentage", params["percentage"])

    rsg = NeuralNetRegressor(lr = params["lr"],
                             optimizer__weight_decay = params["optimizer__weight_decay"],
                             criterion = params["criterion"],
                             batch_size = params["batch_size"],
                             optimizer__betas = params["optimizer__betas"],
                             module = create_nn_module(params["input_nodes"], params["hidden_layers"], 
                                                       params["hidden_nodes"], params["output_nodes"], params["percentage"]),
                             max_epochs = params["max_epochs"],
                             callbacks=[skorch.callbacks.EarlyStopping(patience=params["patience"])],
                             device = params["device"],
                             optimizer = params["optimizer"]
                            )
    init_module(rsg.module, params["weight_mode"], params["bias"])
    
    #这里好像是无法使用skf的呀，不对只是新的skf需要其他设置啊，需要修改Y_train的shape咯
    #skf = StratifiedKFold(Y_train, n_folds=5, shuffle=True, random_state=None)
    #这里sklearn的均方误差是可以为负数的，我还以为是自己的代码出现了问题了呢
    metric = cross_val_score(rsg, X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.float32), cv=8, scoring="neg_mean_squared_log_error").mean()
    #metric = cross_val_score(rsg, X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.float32), cv=2, scoring="neg_mean_squared_error").mean()
    print(metric)
    return -metric

def parse_nn_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes = {}
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    best_nodes["batch_size"] = space_nodes["batch_size"][trials_list[0]["misc"]["vals"]["batch_size"][0]]
    best_nodes["criterion"] = space_nodes["criterion"][trials_list[0]["misc"]["vals"]["criterion"][0]]
    best_nodes["max_epochs"] = space_nodes["max_epochs"][trials_list[0]["misc"]["vals"]["max_epochs"][0]]

    best_nodes["lr"] = space_nodes["lr"][trials_list[0]["misc"]["vals"]["lr"][0]] 
    best_nodes["optimizer__betas"] = space_nodes["optimizer__betas"][trials_list[0]["misc"]["vals"]["optimizer__betas"][0]]
    best_nodes["optimizer__weight_decay"] = space_nodes["optimizer__weight_decay"][trials_list[0]["misc"]["vals"]["optimizer__weight_decay"][0]]
    best_nodes["weight_mode"] = space_nodes["weight_mode"][trials_list[0]["misc"]["vals"]["weight_mode"][0]]
    best_nodes["bias"] = space_nodes["bias"][trials_list[0]["misc"]["vals"]["bias"][0]]
    best_nodes["patience"] = space_nodes["patience"][trials_list[0]["misc"]["vals"]["patience"][0]]
    best_nodes["device"] = space_nodes["device"][trials_list[0]["misc"]["vals"]["device"][0]]
    best_nodes["optimizer"] = space_nodes["optimizer"][trials_list[0]["misc"]["vals"]["optimizer"][0]]
    
    #新添加的这些元素用于控制模型的结构
    best_nodes["input_nodes"] = space_nodes["input_nodes"][trials_list[0]["misc"]["vals"]["input_nodes"][0]]
    best_nodes["hidden_layers"] = space_nodes["hidden_layers"][trials_list[0]["misc"]["vals"]["hidden_layers"][0]]
    best_nodes["hidden_nodes"] = space_nodes["hidden_nodes"][trials_list[0]["misc"]["vals"]["hidden_nodes"][0]]
    best_nodes["output_nodes"] = space_nodes["output_nodes"][trials_list[0]["misc"]["vals"]["output_nodes"][0]]
    best_nodes["percentage"] = space_nodes["percentage"][trials_list[0]["misc"]["vals"]["percentage"][0]]

    return best_nodes

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)    

def cal_rmse(Y_train_pred, Y_train):
    
    #当遇到nan的时候所有的数据都会变成nan
    error = Y_train_pred- Y_train
    error = torch.from_numpy(error)
    return float(torch.sqrt(torch.mean(error*error)))
    
def cal_nnrsg_rmse(rsg, X_train, Y_train):
    
    Y_train_pred = rsg.predict(X_train.astype(np.float32))
    return cal_rmse(Y_train_pred, Y_train)

def record_best_model_rmse(rsg, rsme, best_model, best_rmse):
    
    flag = False
    
    if not isclose(best_rmse, rsme):
        if best_rmse > rsme:
            flag = True
            best_rmse = rsme
            best_model = rsg
            
    return best_model, best_rmse, flag

def train_nn_model_validate1(nodes, X_train_scaled, Y_train, max_evals=10):
    
    #我觉得0.12的设置有点多了，还有很多数据没用到呢，感觉这样子设置应该会好一些的吧？
    #X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.12, stratify=Y_train)
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.14)
    #由于神经网络模型初始化、dropout等的问题导致网络不够稳定
    #解决这个问题的办法就是多重复计算几次，选择其中靠谱的模型
    best_rmse = 99999999999.9
    best_model = 0.0
    for j in range(0, max_evals):
        
        rsg = NeuralNetRegressor(lr = nodes["lr"],
                                 optimizer__weight_decay = nodes["optimizer__weight_decay"],
                                 criterion = nodes["criterion"],
                                 batch_size = nodes["batch_size"],
                                 optimizer__betas = nodes["optimizer__betas"],
                                 module = create_nn_module(nodes["input_nodes"], nodes["hidden_layers"], 
                                                           nodes["hidden_nodes"], nodes["output_nodes"], nodes["percentage"]),
                                 max_epochs = nodes["max_epochs"],
                                 callbacks=[skorch.callbacks.EarlyStopping(patience=nodes["patience"])],
                                 device = nodes["device"],
                                 optimizer = nodes["optimizer"]
                                 )
        init_module(rsg.module, nodes["weight_mode"], nodes["bias"])
        rsg.fit(X_split_train.astype(np.float32), Y_split_train.astype(np.float32))
            
        #Y_pred = rsg.predict(X_split_test.astype(np.float32))
        metric = cal_nnrsg_rmse(rsg, X_split_test, Y_split_test)
        
        best_model, best_rmse, flag = record_best_model_rmse(rsg, metric, best_model, best_rmse)        
    
    return best_model, best_rmse

def nn_predict(best_nodes, X_train_scaled, Y_train, X_test_scaled, max_evals):
    
    best_model, best_rmse = train_nn_model_validate1(best_nodes, X_train_scaled.values, Y_train.values, max_evals)
    Y_pred = best_model.predict(X_test_scaled.values.astype(np.float32))
    T_pred = best_model.predict(X_train_scaled.values.astype(np.float32))
    
    #输入mse和rmse的计算结果
    print("mse:", np.mean((np.expm1(T_pred)-np.expm1(Y_train))**2))
    print("rmse:", np.sqrt(np.mean((np.expm1(T_pred)-np.expm1(Y_train))**2)))

    #这边并不是需要对data_test进行预测和写入文件
    Y_pred = Y_pred.flatten()
    data = {"id":data_test["id"], "revenue":np.expm1(Y_pred)}
    output = pd.DataFrame(data = data)        
    #output.to_csv(best_nodes["path"], index=False)
    output.to_csv("nn_predicton.csv", index=False)

nn_space = {"title":hp.choice("title", ["stacked_tmdb_box_office_prediction"]),
            "path":hp.choice("path", ["TMDB_Box_Office_Prediction.csv"]),
            "mean":hp.choice("mean", [0]),
            "std":hp.choice("std", [0]),
            "max_epochs":hp.choice("max_epochs",[3000]),
            "patience":hp.choice("patience", [3,6,9]),
            "lr":hp.choice("lr", [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.00010,
                                  0.00011, 0.00012, 0.00013, 0.00014, 0.00015, 0.00016, 0.00017, 0.00018, 0.00019, 0.00020,
                                  0.00021, 0.00022, 0.00023, 0.00024, 0.00025, 0.00026, 0.00027, 0.00028, 0.00029, 0.00030,
                                  0.00031, 0.00032, 0.00033, 0.00034, 0.00035, 0.00036, 0.00037, 0.00038, 0.00039, 0.00040,
                                  0.00041, 0.00042, 0.00043, 0.00044, 0.00045, 0.00046, 0.00047, 0.00048, 0.00049, 0.00050,
                                  0.00051, 0.00052, 0.00053, 0.00054, 0.00055, 0.00056, 0.00057, 0.00058, 0.00059, 0.00060,
                                  0.00061, 0.00062, 0.00063, 0.00064, 0.00065, 0.00066, 0.00067, 0.00068, 0.00069, 0.00070,
                                  0.00071, 0.00072, 0.00073, 0.00074, 0.00075, 0.00076, 0.00077, 0.00078, 0.00079, 0.00080,
                                  0.00081, 0.00082, 0.00083, 0.00084, 0.00085, 0.00086, 0.00087, 0.00088, 0.00089, 0.00090,
                                  0.00091, 0.00092, 0.00093, 0.00094, 0.00095, 0.00096, 0.00097, 0.00098, 0.00099, 0.00100,
                                  0.00101, 0.00102, 0.00103, 0.00104, 0.00105, 0.00106, 0.00107, 0.00108, 0.00109, 0.00110,
                                  0.00111, 0.00112, 0.00113, 0.00114, 0.00115, 0.00116, 0.00117, 0.00118, 0.00119, 0.00120,
                                  0.00121, 0.00122, 0.00123, 0.00124, 0.00125, 0.00126, 0.00127, 0.00128, 0.00129, 0.00130,
                                  0.00131, 0.00132, 0.00133, 0.00134, 0.00135, 0.00136, 0.00137, 0.00138, 0.00139, 0.00140,
                                  0.00141, 0.00142, 0.00143, 0.00144, 0.00145, 0.00146, 0.00147, 0.00148, 0.00149, 0.00150,
                                  0.00151, 0.00152, 0.00153, 0.00154, 0.00155, 0.00156, 0.00157, 0.00158, 0.00159, 0.00160,
                                  0.00161, 0.00162, 0.00163, 0.00164, 0.00165, 0.00166, 0.00167, 0.00168, 0.00169, 0.00170,
                                  0.00171, 0.00172, 0.00173, 0.00174, 0.00175, 0.00176, 0.00177, 0.00178, 0.00179, 0.00180]),  
            "optimizer__weight_decay":hp.choice("optimizer__weight_decay",[0.000]),  
            "criterion":hp.choice("criterion", [RMSELoss]),

            "batch_size":hp.choice("batch_size", [128]),
            "optimizer__betas":hp.choice("optimizer__betas",[[0.90, 0.9999]]),
            "input_nodes":hp.choice("input_nodes", [385]),
            "hidden_layers":hp.choice("hidden_layers", [1, 3, 5, 7, 9, 11]), 
            "hidden_nodes":hp.choice("hidden_nodes", [300, 350, 400, 450, 500, 550, 600, 650, 700, 
                                                   750, 800, 850, 900, 950, 1000, 1050, 1100]), 
            "output_nodes":hp.choice("output_nodes", [1]),
            "percentage":hp.choice("percentage", [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]),
            "weight_mode":hp.choice("weight_mode", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),

            "bias":hp.choice("bias", [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]),
            "device":hp.choice("device", ["cuda"]),
            "optimizer":hp.choice("optimizer", [torch.optim.Adam])
            }

nn_space_nodes = {"title":["stacked_tmdb_box_office_prediction"],
                  "path":["TMDB_Box_Office_Prediction.csv"],
                  "mean":[0],
                  "std":[0],
                  "max_epochs":[3000],
                  "patience":[3,6,9],
                  "lr":[0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.00010,
                        0.00011, 0.00012, 0.00013, 0.00014, 0.00015, 0.00016, 0.00017, 0.00018, 0.00019, 0.00020,
                        0.00021, 0.00022, 0.00023, 0.00024, 0.00025, 0.00026, 0.00027, 0.00028, 0.00029, 0.00030,
                        0.00031, 0.00032, 0.00033, 0.00034, 0.00035, 0.00036, 0.00037, 0.00038, 0.00039, 0.00040,
                        0.00041, 0.00042, 0.00043, 0.00044, 0.00045, 0.00046, 0.00047, 0.00048, 0.00049, 0.00050,
                        0.00051, 0.00052, 0.00053, 0.00054, 0.00055, 0.00056, 0.00057, 0.00058, 0.00059, 0.00060,
                        0.00061, 0.00062, 0.00063, 0.00064, 0.00065, 0.00066, 0.00067, 0.00068, 0.00069, 0.00070,
                        0.00071, 0.00072, 0.00073, 0.00074, 0.00075, 0.00076, 0.00077, 0.00078, 0.00079, 0.00080,
                        0.00081, 0.00082, 0.00083, 0.00084, 0.00085, 0.00086, 0.00087, 0.00088, 0.00089, 0.00090,
                        0.00091, 0.00092, 0.00093, 0.00094, 0.00095, 0.00096, 0.00097, 0.00098, 0.00099, 0.00100,
                        0.00101, 0.00102, 0.00103, 0.00104, 0.00105, 0.00106, 0.00107, 0.00108, 0.00109, 0.00110,
                        0.00111, 0.00112, 0.00113, 0.00114, 0.00115, 0.00116, 0.00117, 0.00118, 0.00119, 0.00120,
                        0.00121, 0.00122, 0.00123, 0.00124, 0.00125, 0.00126, 0.00127, 0.00128, 0.00129, 0.00130,
                        0.00131, 0.00132, 0.00133, 0.00134, 0.00135, 0.00136, 0.00137, 0.00138, 0.00139, 0.00140,
                        0.00141, 0.00142, 0.00143, 0.00144, 0.00145, 0.00146, 0.00147, 0.00148, 0.00149, 0.00150,
                        0.00151, 0.00152, 0.00153, 0.00154, 0.00155, 0.00156, 0.00157, 0.00158, 0.00159, 0.00160,
                        0.00161, 0.00162, 0.00163, 0.00164, 0.00165, 0.00166, 0.00167, 0.00168, 0.00169, 0.00170,
                        0.00171, 0.00172, 0.00173, 0.00174, 0.00175, 0.00176, 0.00177, 0.00178, 0.00179, 0.00180],
                  "optimizer__weight_decay":[0.000],
                  "criterion":[RMSELoss],
                  #这个参数使用固定值主要是考虑计算时间
                  "batch_size":[128],
                  #这个参数使用默认参数能够减少超参搜索范围，从而获得更加结果？
                  #但是就这个练手项目而言，就暂时先是这个样子了吧。
                  "optimizer__betas":[[0.90, 0.999]],
                  "input_nodes":[385],
                  "hidden_layers":[1, 3, 5, 7, 9, 11], 
                  "hidden_nodes":[300, 350, 400, 450, 500, 550, 600, 650, 700, 
                                  750, 800, 850, 900, 950, 1000, 1050, 1100], 
                  "output_nodes":[1],
                  "percentage":[0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
                  "weight_mode":[1, 2, 3, 4, 5, 6, 7,
                                 8, 9, 10, 11, 12, 13],
                  "bias":[-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03],
                  "device":["cuda"],
                  "optimizer":[torch.optim.Adam]
                  }

#lgb相关的部分
def lgb_f(params):
                    
    print("title", params["title"])
    print("path", params["path"])
    print("mean", params["mean"])
    print("std", params["std"])
    print("learning_rate", params["learning_rate"])
    print("n_estimators", params["n_estimators"])
    print("max_depth", params["max_depth"])
    #print("eval_metric", params["eval_metric"])
    print("num_leaves", params["num_leaves"])
    print("subsample", params["subsample"])
    print("colsample_bytree", params["colsample_bytree"])
    print("min_child_samples", params["min_child_samples"])
    print("min_child_weight", params["min_child_weight"])   

    rsg = LGBMRegressor(learning_rate=params["learning_rate"],
                        n_estimators=int(params["n_estimators"]),
                        max_depth=params["max_depth"],
                        #eval_metric=params["eval_metric"],
                        num_leaves=params["num_leaves"],
                        subsample=params["subsample"],
                        colsample_bytree=params["colsample_bytree"],
                        min_child_samples=params["min_child_samples"],
                        min_child_weight=params["min_child_weight"])

    #skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(rsg, X_train_scaled, Y_train, cv=10, scoring="neg_mean_squared_error").mean()
    
    print(-metric)
    #主要我想看rmse的结果，但是metric的结果是负数
    #所以采用下面的形式返回
    print((-metric)**0.5) 
    print()
    return -metric
    
def parse_lgb_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes = {}
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
        
    best_nodes["learning_rate"] = space_nodes["learning_rate"][trials_list[0]["misc"]["vals"]["learning_rate"][0]]
    best_nodes["n_estimators"] = space_nodes["n_estimators"][trials_list[0]["misc"]["vals"]["n_estimators"][0]]
    best_nodes["max_depth"] = space_nodes["max_depth"][trials_list[0]["misc"]["vals"]["max_depth"][0]]
    #best_nodes["eval_metric"]= space_nodes["eval_metric"][trials_list[0]["misc"]["vals"]["eval_metric"][0]]
    best_nodes["num_leaves"] = space_nodes["num_leaves"][trials_list[0]["misc"]["vals"]["num_leaves"][0]]
    best_nodes["subsample"] = space_nodes["subsample"][trials_list[0]["misc"]["vals"]["subsample"][0]]
    best_nodes["colsample_bytree"] = space_nodes["colsample_bytree"][trials_list[0]["misc"]["vals"]["colsample_bytree"][0]]
    best_nodes["min_child_samples"] = space_nodes["min_child_samples"][trials_list[0]["misc"]["vals"]["min_child_samples"][0]]
    best_nodes["min_child_weight"]= space_nodes["min_child_weight"][trials_list[0]["misc"]["vals"]["min_child_weight"][0]]
    
    return best_nodes

def train_lgb_model(best_nodes, X_train_scaled, Y_train):
    
    rsg = LGBMRegressor(learning_rate=best_nodes["learning_rate"],
                        n_estimators=int(best_nodes["n_estimators"]),
                        max_depth=best_nodes["max_depth"],
                        #eval_metric=best_nodes["eval_metric"],
                        num_leaves=best_nodes["num_leaves"],
                        subsample=best_nodes["subsample"],
                        colsample_bytree=best_nodes["colsample_bytree"],
                        min_child_samples=best_nodes["min_child_samples"],
                        min_child_weight=best_nodes["min_child_weight"])

    rsg.fit(X_train_scaled, Y_train)
    Y_pred = rsg.predict(X_train_scaled)
    print("mse:", np.mean((Y_pred-Y_train)**2))
    print("rmse:", np.sqrt(np.mean((Y_pred-Y_train)**2)))
    return rsg

#其他的参数暂时不知道如何设置，暂时就用这些超参吧，我觉得主要还是特征的创建咯
lgb_space = {"title":hp.choice("title", ["stacked_tmdb_box_office_prediction"]),
             "path":hp.choice("path", ["TMDB_Box_Office_Prediction.csv"]),
             "mean":hp.choice("mean", [0]),
             "std":hp.choice("std", [0]),
             "learning_rate":hp.choice("learning_rate", np.linspace(0.01, 0.40, 40)),
             "n_estimators":hp.choice("n_estimators", np.linspace(60, 200, 8)),
             "max_depth":hp.choice("max_depth", [-1, 3, 4, 6, 8]),
             #"eval_metric":hp.choice("eval_metric", ["l2_root"]),
             "num_leaves":hp.choice("num_leaves", [21, 31, 54]),
             "subsample":hp.choice("subsample", [0.8, 0.9, 1.0]),
             "colsample_bytree":hp.choice("colsample_bytree", [0.8, 0.9, 1.0]),
             "min_child_samples":hp.choice("min_child_samples", [16, 18, 20, 22, 24]),
             "min_child_weight":hp.choice("min_child_weight", [16, 18, 20, 22, 24])
             #"reg_alpha":hp.choice("reg_alpha", [0, 0.001, 0.01, 0.03, 0.1, 0.3]),
             #"reg_lambda":hp.choice("reg_lambda", [0, 0.001, 0.01, 0.03, 0.1, 0.3]),
            }

lgb_space_nodes = {"title":["stacked_tmdb_box_office_prediction"],
                   "path":["TMDB_Box_Office_Prediction.csv"],
                   "mean":[0],
                   "std":[0],
                   "learning_rate":np.linspace(0.01, 0.40, 40),
                   "n_estimators":np.linspace(60, 200, 8),
                   "max_depth":[-1, 3, 4, 6, 8],
                   #"eval_metric":["l2_root"],
                   "num_leaves":[21, 31, 54],
                   "subsample":[0.8, 0.9, 1.0],
                   "colsample_bytree":[0.8, 0.9, 1.0],
                   "min_child_samples":[16, 18, 20, 22, 24],
                   "min_child_weight":[16, 18, 20, 22, 24]
                   }

"""
#xgb和cat略微PK了一下，感觉后者潜力无限的样子，好像能吊打xgb
rsg = XGBRegressor(random_state=42)
rsg.fit(X_train_scaled, Y_train)
Y_pred = rsg.predict(X_train_scaled)
print("mse:", np.mean((Y_pred-Y_train)**2))
print("rmse:", np.sqrt(np.mean((Y_pred-Y_train)**2)))
Y_pred = rsg.predict(X_test_scaled)
data = {"id":data_test["id"], "revenue":Y_pred}
output = pd.DataFrame(data = data)
output.to_csv("xgb_predicton0.csv", index=False)

#下面的设置方式可以使用GPU对于数据进行训练咯
rsg = CatBoostRegressor(random_state=42, task_type='GPU', iterations=20000)
rsg.fit(X_train_scaled, Y_train)
Y_pred = rsg.predict(X_train_scaled)
print("mse:", np.mean((Y_pred-Y_train)**2))
print("rmse:", np.sqrt(np.mean((Y_pred-Y_train)**2)))
Y_pred = rsg.predict(X_test_scaled)
data = {"id":data_test["id"], "revenue":Y_pred}
output = pd.DataFrame(data = data)
output.to_csv("cat_predicton0.csv", index=False)
"""

"""
#catboost的超参搜索咯
#因为CatBoostRegressor没有coef_所以无法使用PermutationImportance进行特征选择咯，就使用xgb代劳咯
#我的天不对，CatBoostRegressor可以使用PermutationImportance进行特征选择，但是巨慢，而且选择结果似乎有问题
#这样我晚上测试一哈能否直接用cat进行特征的选择，如果不行的话就直接用xgb代劳了吧。
#下面的代码可以正常运行，就是结果好像并不和预期一致熬
start_time = datetime.datetime.now()
rfc_model = CatBoostRegressor(random_state=42, task_type='GPU').fit(X_train_scaled, Y_train)
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
best = fmin(cat_f, cat_space, algo=tpe.suggest, max_evals=1, trials=trials)
best_nodes = parse_cat_nodes(trials, cat_space_nodes)
save_inter_params(trials, cat_space_nodes, best_nodes, "tmdb_box_office_prediction")
rsg = train_cat_model(best_nodes, X_train_scaled, Y_train)

Y_pred = rsg.predict(X_test_scaled)
data = {"id":data_test["id"], "revenue":Y_pred}
output = pd.DataFrame(data = data)            
output.to_csv("cat_predicton.csv", index=False)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

"""
#现在正在实验不经过特征选择的时候结果是否更好？
#这个实验证明了，xgb模型特征选择前后好像差距不大，总体而言略有提升。
#这个实验证明了，cat特征选择之后的模型效果嗨不如没经过特征选择的模型
#这个可能是因为模型本身的设计问题，cat会组合特征的，所以不应该这样选择特征吧
#而且这个实验也算是可以将cat模型和xgb模型进行了一哈比较,cat就是速度慢但是比xgb效果更好。
#mse: 4494326966815931.0
#rmse: 67039741.697115235
#-7031069324633610.0
#-5970638786253380.0
#mse: 4494326966815931.0
#rmse: 67039741.697115235
#-6876736596297556.0
#-5688557476764342.0
X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.3, random_state=42)
xgb = XGBRegressor(random_state=42)
xgb.fit(X_split_train, Y_split_train)
Y_pred = xgb.predict(X_split_test)
print("mse:", np.mean((Y_pred-Y_split_test)**2))
print("rmse:", np.sqrt(np.mean((Y_pred-Y_split_test)**2)))
metric1 = cross_val_score(xgb, X_split_train, Y_split_train, cv=10, scoring="neg_mean_squared_error").mean()
metric2 = cross_val_score(xgb, X_split_test, Y_split_test, cv=10, scoring="neg_mean_squared_error").mean()
print(metric1)
print(metric2)

start_time = datetime.datetime.now()
rfc_model = XGBRegressor(random_state=42).fit(X_split_train, Y_split_train)
perm = PermutationImportance(rfc_model, random_state=42).fit(X_split_train, Y_split_train)
feature_importances1 = perm.feature_importances_#这是返回每个特征的权重
feature_importances_std = perm.feature_importances_std_ 
feature_importances2 = np.where(feature_importances1>0)#此时我记录下了每个特征的列数
X_split_train_new = X_split_train[X_split_train.columns[feature_importances2]]
X_split_test_new = X_split_test[X_split_test.columns[feature_importances2]]
X_split_train = X_split_train_new
X_split_test = X_split_test_new
xgb = XGBRegressor(random_state=42)
xgb.fit(X_split_train, Y_split_train)
Y_pred = xgb.predict(X_split_test)
print("mse:", np.mean((Y_pred-Y_split_test)**2))
print("rmse:", np.sqrt(np.mean((Y_pred-Y_split_test)**2)))
metric1 = cross_val_score(xgb, X_split_train, Y_split_train, cv=10, scoring="neg_mean_squared_error").mean()
metric2 = cross_val_score(xgb, X_split_test, Y_split_test, cv=10, scoring="neg_mean_squared_error").mean()
print(metric1)
print(metric2)


mse: 5234728273565622.0
rmse: 72351422.05627766
-6936366034980786.0
-5721180417973920.0
mse: 1.5923488227388196e+16
rmse: 126188304.63790293
-2.012557271674441e+16
-1.6076423120921994e+16
X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.3, random_state=42)
cat = CatBoostRegressor(random_state=42, iterations=6000, task_type='GPU')
cat.fit(X_split_train, Y_split_train)
Y_pred = cat.predict(X_split_test)
print("mse:", np.mean((Y_pred-Y_split_test)**2))
print("rmse:", np.sqrt(np.mean((Y_pred-Y_split_test)**2)))
metric1 = cross_val_score(cat, X_split_train, Y_split_train, cv=10, scoring="neg_mean_squared_error").mean()
metric2 = cross_val_score(cat, X_split_test, Y_split_test, cv=10, scoring="neg_mean_squared_error").mean()
print(metric1)
print(metric2)

start_time = datetime.datetime.now()
rfc_model = CatBoostRegressor(random_state=42, iterations=6000, task_type='GPU').fit(X_split_train, Y_split_train)
perm = PermutationImportance(rfc_model, random_state=42).fit(X_split_train, Y_split_train)
feature_importances1 = perm.feature_importances_#这是返回每个特征的权重
feature_importances_std = perm.feature_importances_std_ 
feature_importances2 = np.where(feature_importances1>0)#此时我记录下了每个特征的列数
X_split_train_new = X_split_train[X_split_train.columns[feature_importances2]]
X_split_test_new = X_split_test[X_split_test.columns[feature_importances2]]
X_split_train = X_split_train_new
X_split_test = X_split_test_new
cat = CatBoostRegressor(random_state=42, iterations=6000, task_type='GPU')
cat.fit(X_split_train, Y_split_train)
Y_pred = cat.predict(X_split_test)
print("mse:", np.mean((Y_pred-Y_split_test)**2))
print("rmse:", np.sqrt(np.mean((Y_pred-Y_split_test)**2)))
metric1 = cross_val_score(cat, X_split_train, Y_split_train, cv=10, scoring="neg_mean_squared_error").mean()
metric2 = cross_val_score(cat, X_split_test, Y_split_test, cv=10, scoring="neg_mean_squared_error").mean()
print(metric1)
print(metric2)
"""

"""
#这个实验说明iterations取1000真的比取3000和6000的好咯
5037457517447120.0
70975048.55544038
-7038764562393806.0
-5606052626595302.0

5215102693194729.0
72215667.92043628
-7172024191322267.0
-5818345579568044.0

5295539335032408.0
72770456.47123843
-7212143272887496.0
-5892317033428227.0
result = []
X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.3, random_state=42)
cat = CatBoostRegressor(random_state=42, iterations=1000, task_type='GPU')
cat.fit(X_split_train, Y_split_train)
Y_pred = cat.predict(X_split_test)
mse = np.mean((Y_pred-Y_split_test)**2)
rmse = np.sqrt(np.mean((Y_pred-Y_split_test)**2))
print("mse:", mse)
print("rmse:", rmse)
metric1 = cross_val_score(cat, X_split_train, Y_split_train, cv=10, scoring="neg_mean_squared_error").mean()
metric2 = cross_val_score(cat, X_split_test, Y_split_test, cv=10, scoring="neg_mean_squared_error").mean()
print(metric1)
print(metric2)
result.append(mse)
result.append(rmse)
result.append(metric1)
result.append(metric2)

cat = CatBoostRegressor(random_state=42, iterations=3000, task_type='GPU')
cat.fit(X_split_train, Y_split_train)
Y_pred = cat.predict(X_split_test)
mse = np.mean((Y_pred-Y_split_test)**2)
rmse = np.sqrt(np.mean((Y_pred-Y_split_test)**2))
print("mse:", mse)
print("rmse:", rmse)
metric1 = cross_val_score(cat, X_split_train, Y_split_train, cv=10, scoring="neg_mean_squared_error").mean()
metric2 = cross_val_score(cat, X_split_test, Y_split_test, cv=10, scoring="neg_mean_squared_error").mean()
print(metric1)
print(metric2)
result.append(mse)
result.append(rmse)
result.append(metric1)
result.append(metric2)

cat = CatBoostRegressor(random_state=42, iterations=6000, task_type='GPU')
cat.fit(X_split_train, Y_split_train)
Y_pred = cat.predict(X_split_test)
mse = np.mean((Y_pred-Y_split_test)**2)
rmse = np.sqrt(np.mean((Y_pred-Y_split_test)**2))
print("mse:", mse)
print("rmse:", rmse)
metric1 = cross_val_score(cat, X_split_train, Y_split_train, cv=10, scoring="neg_mean_squared_error").mean()
metric2 = cross_val_score(cat, X_split_test, Y_split_test, cv=10, scoring="neg_mean_squared_error").mean()
print(metric1)
print(metric2)
result.append(mse)
result.append(rmse)
result.append(metric1)
result.append(metric2)

for i in range(0, len(result)):
    print(result[i])
"""

"""
#下面的代码可以运行了，但是效果真的很差劲熬
#下面的前两行是cat未优化模型的结果
#后两行是执行三次超参搜索的神经网络模型的结果
#mse: 5234728273565622.0
#rmse: 72351422.05627766
#mse: revenue    1.948313e+16
#rmse: revenue    139581984.0
Y_train_temp = Y_train.values.reshape(-1,1)
Y_train_temp = np.log1p(Y_train_temp)
Y_train = pd.DataFrame(data=Y_train_temp.astype(np.float32), columns=["revenue"])
start_time = datetime.datetime.now()
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best_params = fmin(nn_f, nn_space, algo=algo, max_evals=1, trials=trials)
best_nodes = parse_nn_nodes(trials, nn_space_nodes)
save_inter_params(trials, nn_space_nodes, best_nodes, "tmdb_box_office_prediction")
nn_predict(best_nodes, X_train_scaled, Y_train, X_test_scaled, 5)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

"""
#下面的lgb的超参搜索终于也可以使用咯
start_time = datetime.datetime.now()
rfc_model = LGBMRegressor(random_state=42).fit(X_train_scaled, Y_train)
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
best = fmin(lgb_f, lgb_space, algo=tpe.suggest, max_evals=1, trials=trials)
best_nodes = parse_lgb_nodes(trials, lgb_space_nodes)
save_inter_params(trials, lgb_space_nodes, best_nodes, "tmdb_box_office_prediction")
rsg = train_lgb_model(best_nodes, X_train_scaled, Y_train)

Y_pred = rsg.predict(X_test_scaled)
data = {"id":data_test["id"], "revenue":Y_pred}
output = pd.DataFrame(data = data)
output.to_csv("lgb_predicton.csv", index=False)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

"""
#这个实验证明了，lgb模型特征选择前后好像差距不大，总体而言略有提升。
mse: 4507820707869479.0
rmse: 67140306.1347614
-6728892001288328.0
-5786222217401582.0
mse: 4507820707869479.0
rmse: 67140306.1347614
-6726981237052374.0
-5744437712744022.0
start_time = datetime.datetime.now()
X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.3, random_state=42)
lgb = LGBMRegressor(random_state=42)
lgb.fit(X_split_train, Y_split_train)
Y_pred = lgb.predict(X_split_test)
print("mse:", np.mean((Y_pred-Y_split_test)**2))
print("rmse:", np.sqrt(np.mean((Y_pred-Y_split_test)**2)))
metric1 = cross_val_score(lgb, X_split_train, Y_split_train, cv=10, scoring="neg_mean_squared_error").mean()
metric2 = cross_val_score(lgb, X_split_test, Y_split_test, cv=10, scoring="neg_mean_squared_error").mean()
print(metric1)
print(metric2)

rfc_model = LGBMRegressor(random_state=42).fit(X_split_train, Y_split_train)
perm = PermutationImportance(rfc_model, random_state=42).fit(X_split_train, Y_split_train)
feature_importances1 = perm.feature_importances_#这是返回每个特征的权重
feature_importances_std = perm.feature_importances_std_ 
feature_importances2 = np.where(feature_importances1>0)#此时我记录下了每个特征的列数
X_split_train_new = X_split_train[X_split_train.columns[feature_importances2]]
X_split_test_new = X_split_test[X_split_test.columns[feature_importances2]]
X_split_train = X_split_train_new
X_split_test = X_split_test_new
lgb = LGBMRegressor(random_state=42)
lgb.fit(X_split_train, Y_split_train)
Y_pred = lgb.predict(X_split_test)
print("mse:", np.mean((Y_pred-Y_split_test)**2))
print("rmse:", np.sqrt(np.mean((Y_pred-Y_split_test)**2)))
metric1 = cross_val_score(lgb, X_split_train, Y_split_train, cv=10, scoring="neg_mean_squared_error").mean()
metric2 = cross_val_score(lgb, X_split_test, Y_split_test, cv=10, scoring="neg_mean_squared_error").mean()
print(metric1)
print(metric2)
"""
#coding=utf-8
#这个版本总结了一下别人的kernel，并将之前整理的流程一起用于提交新版本
#目前主要使用lgb作为训练吧，因为他的训练时间确实是比较快速的，提交的时候采用cat模型咯。
#尝试神经网络的特征提取，并用这些模型进行输出咯
import ast
import math
import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
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

#（1）在填充之前先确认了缺失的程度（缺失百分比）
#（2）然后使用均值和众数填充缺失值
#（3）使用skewness，并查看数据的分布或者与待拟合数据关系
#（4）然后对数据进行相关性检查，并对重要缺失数据采用简单模型进行拟合
#（5）然后是各种尝试从现有的特征中创造出新的特征加入到模型中咯
#（6）在输入到学习器之前先进行一次特征选择熬，特征选择和组合不同的模型相差很多（还有使用肉眼选择的），不再统一归纳。
#（7）我觉得神经网络的部分就是自己增加噪音咯，以获取更多的数据咯

#找了几个做的比我更好的人的攻略咯
#https://www.kaggle.com/artgor/eda-feature-engineering-and-model-interpretation
#这里的In[13]的代码的写法好像有点意思熬，原来还可以这样子写程序么
#这里面的In[27]提到的使用top的值而不是直接丢弃应该能够得到更多的信息量（数量、top）
#In[67]还创建了更多的特征？？周三发布的影片居然能够获得更多的报酬？？种类更多收入也容易更高
#我觉得最有效的办法就是自己在网站上直接下载更多倍的数据咯，应该是最无解的做法
#之前好像有个库可以自己无脑创建很多新的特征的吧，能够创建出很多我们平时觉得很匪夷所思的但是有用的特征。
#https://www.kaggle.com/ashishpatel26/now-you-see-me  这个狗币感觉没说任何有用的东西
#https://www.kaggle.com/zero92/tmdb-prediction 这个狗币有点意思的地方在于。。直接使用了别人的数据。
#https://www.kaggle.com/kamalchhirang/eda-feature-engineering-lgb-xgb-cat 
#这狗币import json可以很方便处理json数据咯。
#In[12]的部分我觉得可以用来分析我之后的数据的分布情况。
#对年月日进行一下处理咯
#原来那个神奇的特征是从这里来的呀https://www.kaggle.com/kamalchhirang/eda-feature-engineering-lgb-xgb-cat
#原来那个自动创造特征的东西好像叫做auto feature engineering 
#我现在感觉工作量很大的样子呀。。。。。

#读取数据并合并咯
data_train = pd.read_csv("train.csv", encoding="ANSI")
data_test = pd.read_csv("test.csv", encoding="ANSI")
temp = data_train["revenue"]
data_all = pd.concat([data_train, data_test], axis=0)
data_all = data_all.drop(["id","revenue"], axis=1)
#X_all = data_all.drop(["id","revenue"], axis=1)

#先看一哈总体缺失的内容到底有什么
#个人觉得belongs_to_collection、homepage简单使用。tagline搞不好都可以
#Keywords                  669
#belongs_to_collection    5917
#budget                      0
#cast                       26
#crew                       38
#genres                     23
#homepage                 5032
#imdb_id                     0
#original_language           0
#original_title              0
#overview                   22
#popularity                  0
#poster_path                 2
#production_companies      414
#production_countries      157
#release_date                1
#runtime                     6
#spoken_languages           62
#status                      2
#tagline                  1460
#title                       3
#print(data_all.isnull().sum())

#直接将belongs_to_collection的数据删除了吗，主要感觉不知道怎么用
#这样吧，直接将其替换为有无，这样从信息的角度上看，应该信息更丰富一些
#data_all.drop(["belongs_to_collection"], axis=1)
data_all.loc[(data_all.belongs_to_collection.notnull()), 'belongs_to_collection'] = "not null"
data_all.loc[(data_all.belongs_to_collection.isnull()), 'belongs_to_collection'] = "null"

#处理budget属性就直接用很简单的平均数代替了吧,并不需要用其他值预估咯
#data_all["budget"].fillna(data_all["budget"].mode()[0], inplace=True) 
#均值 np.mean(nums) #中位数 np.median(nums) #众数 np.mode()[0]
data_all["budget"].fillna(data_all["budget"].dropna().mean(), inplace=True)
#因为这个budget并不符合正态分布，所以需要使用np.log进行转换咯
#sns.distplot(data_all["budget"], rug=True)
#plt.show()
#偏度本身有个临界值，有些代码是觉得大于1或者0.6就要进行np.log计算咯
#我好想一直不知道np.log np.log1p np.log10怎么选择，看了下面的数据我心里有底咯
#print(data_all["budget"].skew())
#print(np.log(data_all["budget"]).skew())
#print(np.log1p(data_all["budget"]).skew())
#print(np.log2(data_all["budget"]).skew())
#print(np.log10(data_all["budget"]).skew())
#print(temp.skew())
#print(np.log(temp).skew())
#print(np.log1p(temp).skew())
#print(np.log2(temp).skew())
#print(np.log10(temp).skew())
data_all["budget"] = np.log1p(data_all["budget"])

#genres的特征应该如何处理，感觉好像很麻烦的样子
#由于genres存在为空的情况，所以先将空的填充为[]
#下面整理好了所有的genres下面的类比
#data_all["genres"].fillna("[]", inplace=True)
#看了别人的攻略之后，我发现这里应该添加一个电影种类总数genres_num
data_all["genres"].fillna(data_all["genres"].dropna().mode()[0], inplace=True) 
genres_list = []
for i in range(0, len(data_all)):
    #这边如果不使用iloc那么结果就是错误的，取得的对象是str类型的
    #print(data_all.iloc[i]["genres"])
    #print(i)
    dict_list = ast.literal_eval(data_all.iloc[i]["genres"])
    for j in range(0, len(dict_list)):
        name = dict_list[j]["name"]
        name = str("genres="+name)
        if name not in genres_list:
            genres_list.append(name)
            #print(genres_list)
#根据类别重新构造feature咯
data = np.zeros((len(data_all), len(genres_list)))
genres_df = pd.DataFrame(data, columns=genres_list, index=data_all.index.values)
data = np.zeros((len(data_all), 1))
num_df = pd.DataFrame(data, columns=["genres_num"], index=data_all.index.values)
for i in range(0, len(data_all)):
    dict_list = ast.literal_eval(data_all.iloc[i]["genres"])
    num_df.iloc[i] = len(dict_list)
    for j in range(0, len(dict_list)):
        name = dict_list[j]["name"]
        name = str("genres="+name)
        genres_df.iloc[i][name] = 1
#丢弃data_all中的genres特征，然后合并新的genres_df
data_all = data_all.drop(["genres"], axis=1)
data_all = pd.concat([data_all, genres_df, num_df], axis=1)

#现在考虑处理homepage这个特征咯
data_all.loc[(data_all.homepage.notnull()), 'homepage'] = "not null"
data_all.loc[(data_all.homepage.isnull()), 'homepage'] = "null"
#考虑处理imdb_id这个特征咯
data_all.loc[(data_all.imdb_id.notnull()), 'imdb_id'] = "not null"
data_all.loc[(data_all.imdb_id.isnull()), 'imdb_id'] = "null"
#考虑处理original_language这个特征咯
data_all["original_language"].fillna(data_all["original_language"].dropna().mode()[0], inplace=True)
#考虑处理original_title这个特征咯,暂时不知道咋用直接粗暴使用
data_all.loc[(data_all.original_title.notnull()), 'original_title'] = "not null"
data_all.loc[(data_all.original_title.isnull()), 'original_title'] = "null"

"""
#考虑处理overview这个特征咯,暂时不知道咋用直接粗暴使用
data_all.loc[(data_all.overview.notnull()), 'overview'] = "not null"
data_all.loc[(data_all.overview.isnull()), 'overview'] = "null"
"""
#看了一下别人的kernel我似乎发现了一个更好的用法咯
vectorizer = TfidfVectorizer(
             sublinear_tf=True,
             analyzer='word',
             token_pattern=r'\w{1,}',
             ngram_range=(1, 2),
             min_df=5)
overview_text = vectorizer.fit_transform(data_train['overview'].fillna(''))
#eli5.show_weights(linreg, vec=vectorizer, top=20, feature_filter=lambda x: x != '<BIAS>')
overview_text_df = pd.DataFrame(data=overview_text.toarray(), index=data_all.index.values)
rfc_model = LinearRegression().fit(overview_text_df, np.log1p(temp))
perm = PermutationImportance(rfc_model, random_state=42).fit(overview_text_df, np.log1p(temp))
feature_importances1 = perm.feature_importances_#这是返回每个特征的权重
feature_importances_std = perm.feature_importances_std_ 
feature_importances2 = np.where(feature_importances1>0)#此时我记录下了每个特征的列数
#下面才是真正的把所有的特征都选出来了
overview_text_df_new = overview_text_df[overview_text_df.columns[feature_importances2]]


#考虑处理popularity这个特征咯,暂时不知道咋用直接粗暴使用
data_all["popularity"].fillna(data_all["popularity"].dropna().mean(), inplace=True)
#考虑处理poster_path这个特征咯,暂时不知道咋用直接粗暴使用
data_all.loc[(data_all.poster_path.notnull()), 'poster_path'] = "not null"
data_all.loc[(data_all.poster_path.isnull()), 'poster_path'] = "null"

"""
#考虑处理production_companies这个特征咯,暂时不知道咋用直接粗暴使用
#data_all["production_companies"].fillna("[{'name': 'null', 'id': 4}]", inplace=True)
data_all["production_companies"].fillna(data_all["production_companies"].mode()[0], inplace=True) 
companies_list = []
for i in range(0, len(data_all)):
    dict_list = ast.literal_eval(data_all.iloc[i]["production_companies"])
    for j in range(0, len(dict_list)):
        name = dict_list[j]["name"]
        name = str("production_companies="+name)
        if name not in companies_list:
            companies_list.append(name)
#根据类别重新构造feature咯
data = np.zeros((len(data_all), len(companies_list)))
companies_df = pd.DataFrame(data, columns=companies_list, index=data_all.index.values)
for i in range(0, len(data_all)):
    dict_list = ast.literal_eval(data_all.iloc[i]["production_companies"])
    for j in range(0, len(dict_list)):
        name = dict_list[j]["name"]
        name = str("production_companies="+name)
        companies_df.iloc[i][name] = 1
#丢弃data_all中的genres特征，然后合并新的genres_df
data_all = data_all.drop(["production_companies"], axis=1)
data_all = pd.concat([data_all, companies_df], axis=1)
"""

data_all = data_all.drop(["production_companies"], axis=1)

#考虑处理production_countries这个特征咯
#data_all["production_countries"].fillna("[{'iso_3166_1': 'null', 'name': 'null'}]", inplace=True)
data_all["production_countries"].fillna(data_all["production_countries"].dropna().mode()[0], inplace=True) 
countries_list = []
for i in range(0, len(data_all)):
    dict_list = ast.literal_eval(data_all.iloc[i]["production_countries"])
    for j in range(0, len(dict_list)):
        name = dict_list[j]["iso_3166_1"]
        name = str("production_countries="+name)
        if name not in countries_list:
            countries_list.append(name)
#根据类别重新构造feature咯
data = np.zeros((len(data_all), len(countries_list)))
countries_df = pd.DataFrame(data, columns=countries_list, index=data_all.index.values)
for i in range(0, len(data_all)):
    dict_list = ast.literal_eval(data_all.iloc[i]["production_countries"])
    for j in range(0, len(dict_list)):
        name = dict_list[j]["iso_3166_1"]
        name = str("production_countries="+name)
        countries_df.iloc[i][name] = 1
#丢弃data_all中的genres特征，然后合并新的genres_df
data_all = data_all.drop(["production_countries"], axis=1)
data_all = pd.concat([data_all, countries_df], axis=1)

"""
#考虑处理release_date这个特征咯,我觉得这个特征可以拆分为年和月
data_all["release_date"].fillna(data_all["release_date"].mode()[0], inplace=True) 
years_list = []
months_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
for i in range(0, len(data_all)):
    temp = data_all.iloc[i]["release_date"].split('/')
    #如果是使用年月日的方式表示日期
    if(int(temp[0])>1000): 
        if(temp[0] not in years_list):
            years_list.append(temp[0])
    #如果是使用月日年的方式表示日期
    else:
        if(temp[2] not in years_list):
            if(int(temp[2])<20):
                #如果年份小于20例如15，那么应该是2015年的电影
                year = str("20"+temp[2])
                if(year not in years_list):
                    years_list.append(year)
            else:
                #如果年份大于20例如87，那么应该是1987年的电影
                #但是最早的电影是1895年的，所以如果96可能是1896也可能是1996，但是都只考虑1996吧。。
                year = str("19"+temp[2])
                if(year not in years_list):
                    years_list.append(year)
data = np.zeros((len(data_all), len(years_list)))
years_df = pd.DataFrame(data, columns=years_list, index=data_all.index.values)
data = np.zeros((len(data_all), len(months_list)))
months_df = pd.DataFrame(data, columns=months_list, index=data_all.index.values)
for i in range(0, len(data_all)):
    temp = data_all.iloc[i]["release_date"].split('/')
    #如果是使用年月日的方式表示日期
    if(int(temp[0])>1000): 
        years_df.iloc[i][temp[0]] = 1
        months_df.iloc[i][temp[1]] = 1
    #如果是使用月日年的方式表示日期
    else:    
        if(int(temp[2])<20):
            years_df.iloc[i][str("20" +temp[2])] = 1
            months_df.iloc[i][temp[0]] = 1
        else:
            years_df.iloc[i][str("19"+temp[2])] = 1
            months_df.iloc[i][temp[0]] = 1
data_all = data_all.drop(["release_date"], axis=1)
data_all = pd.concat([data_all, years_df, months_df], axis=1)
"""

#我的天，惊呆我了，excel里面release_date有两种表示方式，但是可以通过excel进行统一设计。
#还好我及时发现了这个问题，但是感觉上面的代码都白写了呀，还是准备重写吧。
#其实我倒是觉得，不一样的日期方式可能暴露了额外的信息哦。
data_all["release_date"].fillna(data_all["release_date"].dropna().mode()[0], inplace=True) 
years_list = []
months_list = ['release_month=1', 'release_month=2', 'release_month=3', 'release_month=4', 
               'release_month=5', 'release_month=6', 'release_month=7', 'release_month=8', 
               'release_month=9', 'release_month=10', 'release_month=11', 'release_month=12']
for i in range(0, len(data_all)):
    temp = data_all.iloc[i]["release_date"].split('/')
    if(str("release_year="+temp[2]) not in years_list):
        years_list.append(str("release_year="+temp[2]))
data = np.zeros((len(data_all), len(years_list)))
years_df = pd.DataFrame(data, columns=years_list, index=data_all.index.values)
data = np.zeros((len(data_all), len(months_list)))
months_df = pd.DataFrame(data, columns=months_list, index=data_all.index.values)        
for i in range(0, len(data_all)):
    temp = data_all.iloc[i]["release_date"].split('/')
    years_df.iloc[i][str("release_year="+temp[2])] = 1
    months_df.iloc[i][str("release_month="+temp[0])] = 1
data_all = data_all.drop(["release_date"], axis=1)
data_all = pd.concat([data_all, years_df, months_df], axis=1)

data_all["runtime"].fillna(data_all["runtime"].dropna().mean(), inplace=True)

#准备处理spoken_languages这个特征咯
#spoken_languages无法被赋值，我真的是想要哭了，为什么这样对我？？
#我的天那，
data_all["spoken_languages"].fillna(data_all["spoken_languages"].dropna().mode()[0], inplace=True)
data_all["spoken_languages"].iloc[150] = "[{'iso_639_1': 'en', 'name': 'English'}]"
languages_list = []
for i in range(0, len(data_all)):
    #print(i)
    dict_list = ast.literal_eval(data_all.iloc[i]["spoken_languages"])
    for j in range(0, len(dict_list)):
        name = dict_list[j]["iso_639_1"]
        name = str("spoken_languages="+name)
        if name not in languages_list:
            languages_list.append(name)
#根据类别重新构造feature咯
data = np.zeros((len(data_all), len(languages_list)))
languages_df = pd.DataFrame(data, columns=languages_list, index=data_all.index.values)
for i in range(0, len(data_all)):
    dict_list = ast.literal_eval(data_all.iloc[i]["spoken_languages"])
    for j in range(0, len(dict_list)):
        name = dict_list[j]["iso_639_1"]
        name = str("spoken_languages="+name)
        languages_df.iloc[i][name] = 1
data_all = data_all.drop(["spoken_languages"], axis=1)
data_all = pd.concat([data_all, languages_df], axis=1)

#考虑处理status这个特征咯,暂时不知道咋用直接粗暴使用
data_all.loc[(data_all.status.isnull()), 'status'] = "null"

#考虑处理tagline这个特征咯,暂时不知道咋用直接粗暴使用
data_all.loc[(data_all.tagline.isnull()), 'tagline'] = "null"
data_all.loc[(data_all.tagline.notnull()), 'tagline'] = "not null"

#考虑处理title这个特征咯,暂时不知道咋用直接粗暴使用
data_all.loc[(data_all.title.isnull()), 'title'] = "null"
data_all.loc[(data_all.title.notnull()), 'title'] = "not null"

"""
#这个也是有一万多个关键词暂时还是不用吧
#考虑处理Keywords这个特征咯,暂时不知道咋用直接粗暴使用
data_all["Keywords"].fillna(data_all["Keywords"].dropna().mode()[0], inplace=True) 
Keywords_list = []
for i in range(0, len(data_all)):
    print(i)
    dict_list = ast.literal_eval(data_all.iloc[i]["Keywords"])
    for j in range(0, len(dict_list)):
        name = dict_list[j]["name"]
        name = str("Keywords="+name)
        if name not in Keywords_list:
            Keywords_list.append(name)
#根据类别重新构造feature咯
data = np.zeros((len(data_all), len(Keywords_list)))
Keywords_df = pd.DataFrame(data, columns=Keywords_list, index=data_all.index.values)
for i in range(0, len(data_all)):
    dict_list = ast.literal_eval(data_all.iloc[i]["Keywords"])
    for j in range(0, len(dict_list)):
        name = dict_list[j]["name"]
        name = str("Keywords="+name)
        Keywords_df.iloc[i][name] = 1
data_all = data_all.drop(["Keywords"], axis=1)
data_all = pd.concat([data_all, Keywords_df], axis=1)
"""

"""
#考虑处理cast这个特征咯,暂时不知道咋用直接粗暴使用
data_all["cast"].fillna(data_all["cast"].mode()[0], inplace=True) 
cast_list = []
for i in range(0, len(data_all)):
    dict_list = ast.literal_eval(data_all.iloc[i]["cast"])
    for j in range(0, len(dict_list)):
        name = dict_list[j]["name"]
        name = str("cast="+name)
        if name not in cast_list:
            cast_list.append(name)
#根据类别重新构造feature咯
data = np.zeros((len(data_all), len(cast_list)))
cast_df = pd.DataFrame(data, columns=cast_list, index=data_all.index.values)
for i in range(0, len(data_all)):
    dict_list = ast.literal_eval(data_all.iloc[i]["cast"])
    for j in range(0, len(dict_list)):
        name = dict_list[j]["name"]
        name = str("cast="+name)
        cast_df.iloc[i][name] = 1
data_all = data_all.drop(["cast"], axis=1)
data_all = pd.concat([data_all, cast_df], axis=1)

#考虑处理crew这个特征咯,暂时不知道咋用直接粗暴使用
data_all["crew"].fillna(data_all["crew"].mode()[0], inplace=True) 
crew_list = []
for i in range(0, len(data_all)):
    dict_list = ast.literal_eval(data_all.iloc[i]["crew"])
    for j in range(0, len(dict_list)):
        name = dict_list[j]["name"]
        name = str("crew="+name)
        if name not in crew_list:
            crew_list.append(name)
#根据类别重新构造feature咯
data = np.zeros((len(data_all), len(crew_list)))
crew_df = pd.DataFrame(data, columns=crew_list, index=data_all.index.values)
for i in range(0, len(data_all)):
    dict_list = ast.literal_eval(data_all.iloc[i]["crew"])
    for j in range(0, len(dict_list)):
        name = dict_list[j]["name"]
        name = str("crew="+name)
        crew_df.iloc[i][name] = 1
data_all = data_all.drop(["crew"], axis=1)
data_all = pd.concat([data_all, crew_df], axis=1)
"""
data_all = data_all.drop(["Keywords"], axis=1)
data_all = data_all.drop(["cast"], axis=1)
data_all = data_all.drop(["crew"], axis=1)

#检查一下经过特征处理以后是否还存在空值
#居然runtime存在空值，所以dataframe的fillna真他吗失效了。。
#1335 2302 243 1489 1632 3817行的数据居然存在空值，我觉得很不可思议
#data_all.isnull().sum(axis=0)可以显示每列为空的数量
print(data_all[data_all.isnull().values==True])

#将数据形成one-hot编码
dict_vector = DictVectorizer(sparse=False)
X_all = data_all
X_all = dict_vector.fit_transform(X_all.to_dict(orient='record'))
X_all = pd.DataFrame(data=X_all, columns=dict_vector.feature_names_)
#将数据形成类似one-hot编码
X_all_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X_all), columns = X_all.columns)
X_all_scaled = pd.DataFrame(data = X_all_scaled, index = X_all.index, columns = X_all_scaled.columns.values)
X_train_scaled = X_all_scaled[:len(data_train)]
X_test_scaled = X_all_scaled[len(data_train):]
Y_train = np.log1p(data_train["revenue"])
#将数据进行存储吧，不然每次经过上述的特征提取实在太花费时间了
X_train_scaled.to_csv("train_scaled.csv", index=False)
X_test_scaled.to_csv("test_scaled.csv", index=False)

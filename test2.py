#coding=utf-8
#看了一下这个比赛的数据，个人觉得这个数据很难再进行补充了，可能会用到one-hot编码之类的东西吧
#太难受了，这个版本主要调试一下data_all["runtime"].fillna(data_all["runtime"].dropna().mean(), inplace=True)
#以及data_all["spoken_languages"].fillna(data_all["spoken_languages"].dropna().mode()[0], inplace=True)
#以及data_all["spoken_languages"].iloc[150] = "[{'iso_639_1': 'en', 'name': 'English'}]"都无法填充为空的元素？？真的难受为什么
#老子心态崩盘了，上面三行代码的问题在家无法解决，在这边不需要修改代码可以直接运行，这是什么情况，难道是dataframe版本不同之类的缘故？？
#excel中的日期存在两种表示方式，可能这两种表示方式暴露了额外的信息，但是目前不打算再利用。
#excel内部可以统一日期表示方式，但是造成了read_csv报错，目前已经采用了设置encoding的方式进行解决
#这个比赛有个大BUG，可以自行使用imdb中的数据，相当于是说，能够搞到更多数据的人自然就可以占据更大的优势
#关了eclipse重新打开好像就可以了，之前家里面的电脑除了关机没进行别的操作
import ast
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectFromModel, RFE, VarianceThreshold

import hyperopt
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

from sklearn.feature_extraction import DictVectorizer

#读取数据并合并咯
data_train = pd.read_csv("train.csv", encoding="ANSI")
data_test = pd.read_csv("test.csv", encoding="ANSI")
temp = data_train["revenue"]
data_all = pd.concat([data_train, data_test], axis=0)
data_all = data_all.drop(["id","revenue"], axis=1)
#X_all = data_all.drop(["id","revenue"], axis=1)

#直接将belongs_to_collection的数据删除了吗，主要感觉不知道怎么用
#这样吧，直接将其替换为有无，这样从信息的角度上看，应该信息更丰富一些
#data_all.drop(["belongs_to_collection"], axis=1)
data_all.loc[(data_all.belongs_to_collection.notnull()), 'belongs_to_collection'] = "not null"
data_all.loc[(data_all.belongs_to_collection.isnull()), 'belongs_to_collection'] = "null"
#处理budget属性就直接用很简单的平均数代替了吧
#data_all["budget"].fillna(data_all["budget"].mode()[0], inplace=True) 
#均值 np.mean(nums) #中位数 np.median(nums) #众数 np.mode()[0]
data_all["budget"].fillna(data_all["budget"].dropna().mean(), inplace=True)

#genres的特征应该如何处理，感觉好像很麻烦的样子
#由于genres存在为空的情况，所以先将空的填充为[]
#下面整理好了所有的genres下面的类比
#data_all["genres"].fillna("[]", inplace=True)
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
for i in range(0, len(data_all)):
    dict_list = ast.literal_eval(data_all.iloc[i]["genres"])
    for j in range(0, len(dict_list)):
        name = dict_list[j]["name"]
        name = str("genres="+name)
        genres_df.iloc[i][name] = 1
#丢弃data_all中的genres特征，然后合并新的genres_df
data_all = data_all.drop(["genres"], axis=1)
data_all = pd.concat([data_all, genres_df], axis=1)

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
#考虑处理overview这个特征咯,暂时不知道咋用直接粗暴使用
data_all.loc[(data_all.overview.notnull()), 'overview'] = "not null"
data_all.loc[(data_all.overview.isnull()), 'overview'] = "null"
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
Y_train = data_train["revenue"]
#将数据进行存储吧，不然每次经过上述的特征提取实在太花费时间了
X_train_scaled.to_csv("train_scaled.csv", index=False)
X_test_scaled.to_csv("test_scaled.csv", index=False)

#接下来是数据相关性的一些分析以及特征选择了吧，我先自己手动清理一些特征吧，再用算法选择特征咯
#也就是先将梳理出来的特征和最后的回报做一些关联吧，不然感觉特征都有尼玛十万个了

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

def lr_f(params):
    
    clf = LogisticRegression(penalty=params["penalty"], C=params["C"],
                             fit_intercept=params["fit_intercept"],
                             class_weight=params["class_weight"])
    selector = RFE(clf, n_features_to_select=params["feature_num"])
    X_train_scaled_new = selector.fit_transform(X_train_scaled, Y_train)
    skf = StratifiedKFold(Y_train, n_folds=10, shuffle=True, random_state=None)
    metric = cross_val_score(clf, X_train_scaled_new, Y_train, cv=skf, scoring="roc_auc").mean()
    return -metric
    
def parse_lr_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes = {}
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    
    best_nodes["feature_num"] = space_nodes["feature_num"][trials_list[0]["misc"]["vals"]["feature_num"][0]]
    best_nodes["penalty"] = space_nodes["penalty"][trials_list[0]["misc"]["vals"]["penalty"][0]]
    best_nodes["C"] = space_nodes["C"][trials_list[0]["misc"]["vals"]["C"][0]]
    best_nodes["fit_intercept"] = space_nodes["fit_intercept"][trials_list[0]["misc"]["vals"]["fit_intercept"][0]]
    best_nodes["class_weight"] = space_nodes["class_weight"][trials_list[0]["misc"]["vals"]["class_weight"][0]]
    
    return best_nodes

def train_lr_model(nodes, X_train_scaled, Y_train):
    
    return 

lr_space = {"title":hp.choice("title", ["stacked_don't_overfit!_II"]),
            "path":hp.choice("path", ["Don't_Overfit!_II_Prediction.csv"]),
            "mean":hp.choice("mean", [0]),
            "std":hp.choice("std", [0]),
            #这个linspace返回的类型是ndarray类型的数据
            "feature_num":hp.choice("feature_num", np.linspace(1,300,300)),
            "penalty":hp.choice("penalty", ["l1", "l2"]),
            "C":hp.choice("C", np.logspace(-3, 5, 100)),
            "fit_intercept":hp.choice("fit_intercept", ["True", "False"]),
            "class_weight":hp.choice("class_weight", ["balanced", None])
            }

lr_space_nodes = {"title":["stacked_don't_overfit!_II"],
                  "path":["Don't_Overfit!_II_Prediction.csv"],
                  "mean":[0],
                  "std":[0],
                  "feature_num":np.linspace(1,300,300),
                  "penalty":["l1", "l2"],
                  "C":np.logspace(-3, 5, 100),
                  "fit_intercept":["True", "False"],
                  "class_weight":["balanced", None]
                 }

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best = fmin(lr_f, lr_space, algo=tpe.suggest, max_evals=8000)
best_nodes = parse_lr_nodes(trials, lr_space_nodes)
save_inter_params(trials, lr_space_nodes, best_nodes, "lr_don't_overfit!_II")


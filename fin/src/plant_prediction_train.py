# -*- coding: utf-8 -*-
# import libraries

import pandas as pd
import numpy as np
import time
import pickle
import os
import json
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
import sklearn.exceptions
from datetime import datetime

import warnings
warnings.filterwarnings('ignore', category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 데이터 셋 (json) 로드 & Dataframe 변환 함수
def load_data(dir):
    dir_path = dir
    column_names = ["fileName", 
                    "GetDateTime", 
                    "ClassID",
                    "Place",
                    "ResultOfGrowthLevel", 
                    "PlantName",
                    "PlantClass",
                    "PlantHeight",
                    "PlantThickness",
                    "RootLength_plant",
                    "RootLength_uproot",
                    "SoilState",
                    "GrowthStage",
                    "Environment",
                    "AirTemperature",
                    "AirHumidity",
                    "Co2",
                    "Quantum",
                    "SupplyEC",
                    "SupplyPH",
                    "HighSoilTemp",
                    "HighSoilHumi",
                    "HighEC",
                    "HighPH",
                    "LowSoilTemp",
                    "LowSoilHumi",
                    "LowSoilEC",
                    "LowSoilPH",
                    "IrrigationState",
                    "WateringTime",
                    "AmtIrrigation",
                    "ImageType",
                    "PhotographerID",
                    "ImageTakeDT",
                    "ImageSize",
                    "ImageName",
                    "FilmingLocation",
                    "ShootingAngle",
                    "ShootingDistance",
                    "ImageResolution",
                    "annotations"]

    column_names_iter = column_names.copy()
    df = pd.DataFrame(columns=column_names)
    column_names_iter.remove("RootLength_plant")
    column_names_iter.remove("RootLength_uproot")
    column_names_iter.append('RootLength')

    dict_tmp_0 = {'fileName': None,
                'GetDateTime': None,
                'ClassID': None,
                'Place': None,
                'ResultOfGrowthLevel': None,
                'PlantName': None,
                'PlantClass': None,
                'PlantHeight': None,
                'PlantThickness': None,
                'RootLength_plant': None,
                'RootLength_uproot': None,
                'SoilState': None,
                'GrowthStage': None,
                'Environment': None,
                'AirTemperature': None,
                'AirHumidity': None,
                'Co2': None,
                'Quantum': None,
                'SupplyEC': None,
                'SupplyPH': None,
                'HighSoilTemp': None,
                'HighSoilHumi': None,
                'HighEC': None,
                'HighPH': None,
                'LowSoilTemp': None,
                'LowSoilHumi': None,
                'LowSoilEC': None,
                'LowSoilPH': None,
                'IrrigationState': None,
                'WateringTime': None,
                'AmtIrrigation': None,
                'ImageType': None,
                'PhotographerID': None,
                'ImageTakeDT': None,
                'ImageSize': None,
                'ImageName': None,
                'FilmingLocation': None,
                'ShootingAngle': None,
                'ShootingDistance': None,
                'ImageResolution': None,
                'annotations': None}

    count = 0
    df_list = []
    f_name_list = []
    for (root, directories, files) in tqdm(list(os.walk(dir_path))):
        for nn, file in enumerate(files):
            if '.json' in file:
                file_path = os.path.join(root, file)
            else:
                continue
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            dict_tmp = dict_tmp_0.copy()
            for mainCategory in json_data.keys():
                if mainCategory != "annotations" :
                    for subCategory in json_data[mainCategory]:
                        if subCategory in column_names_iter:
                            if subCategory == 'RootLength':
                                dict_tmp['RootLength_plant'] = json_data[mainCategory][subCategory]['plant']
                                dict_tmp['RootLength_uproot'] = json_data[mainCategory][subCategory]['uproot']
                            elif subCategory in ['PlantHeight', 'PlantThickness']:
                                if len(json_data[mainCategory][subCategory])>=1:
                                    dict_tmp[subCategory] = list(json_data[mainCategory][subCategory][-1].values())[-1]
                                else:
                                    pass
                            else:
                                dict_tmp[subCategory] = json_data[mainCategory][subCategory]
                else:
                    dict_tmp[mainCategory] = json.dumps(json_data[mainCategory],ensure_ascii=False)
            df_list.append(dict_tmp.copy())
            f_name_list.append(f.name.split('/')[-1])
    df = pd.DataFrame(df_list)
    df = df.replace(np.nan, '', regex=True)
    df['fileName'] = f_name_list
    df = df.reset_index(drop=True).reset_index()
    df['GetDateTime'] = df['GetDateTime'].astype(str)
    df['WateringTime'] = df['WateringTime'].astype(str)
    df['ImageTakeDT'] = df['ImageTakeDT'].astype(str)

    data = df.copy()
    data.drop([0] , axis = 0 , inplace = True)
    print("Loaded dataframe shape ", data.shape)

    return data


OUTPUT_DIR = "/mnt/output"
DATASET_DIR = "/mnt/dataset/생장예측"

# 산출물 버전관리용 파일명 접두사 설정
date_prefix = datetime.now().strftime("%Y-%m-%d")

# 데이터셋별 하위경로 지정
trainset_path = os.path.join(DATASET_DIR, "train", "라벨링데이터")
validset_path = os.path.join(DATASET_DIR, "val", "라벨링데이터")

# 모델 저장경로
model_path = os.path.join(OUTPUT_DIR, f'{date_prefix}_prediction_model.abc')
print(f"trainset경로 {trainset_path},\nvalid set경로 {validset_path},\n모델저장경로 {model_path}")

sttime = datetime.now()


# Train Dataset 불러오기
print('Train Data Loading...')
train_data = load_data(trainset_path)
# Validation Dataset 불러오기
print('Validation Data Loading...')
valid_data = load_data(validset_path)

feature_col_final = ['AirHumidity','AirTemperature','AmtIrrigation','HighEC','HighPH','HighSoilHumi','LowSoilEC',
                     'LowSoilHumi','LowSoilPH','Quantum']

# 주차 컬럼 생성 함수
def week_check(d):
    week_list = [pd.Timestamp(x) for x in ["2023-08-29", "2023-09-05", "2023-09-12", "2023-09-19", "2023-09-26", "2023-10-03", "2023-10-10", "2023-10-17", "2023-10-24", "2023-10-31"]]
    for i in range(11):
        if d < week_list[i]:
            return i
        else:
            pass
    return

# Trainset의 죽은 식물 설정
train_dt_list = ['2023-09-19','2023-09-19','2023-09-19','2023-09-19','2023-09-14','2023-09-19','2023-09-19','2023-09-19','2023-09-14','2023-09-19','2023-10-10','2023-09-14','2023-10-06','2023-09-19','2023-09-19','2023-09-19']
train_class_list = ['A-2-04-L-2','A-2-05-B-1','A-3-08-B-1','A-3-08-B-2','A-3-08-B-3','A-3-08-L-1','A-3-08-L-2','A-3-08-L-3','A-4-10-B-2','A-4-10-L-1','A-4-10-L-1r','A-4-10-L-2','A-4-10-L-2r','A-4-10-L-3','B-1-11-L-1','B-1-11-L-3']
# Validset의 죽은 식물 설정
valid_dt_list = ['2023-10-06','2023-10-07']
valid_class_list = ['A-1-02-L-2','B-1-11-B-1']

# 데이터 전처리 함수
def data_preprocessor(data, dt_list, class_list):
    data_copy = data.copy()
    data_copy['dt'] = pd.to_datetime(data_copy['GetDateTime'], format='%Y%m%d%H%M')

    data_copy['week'] = data_copy['dt'].apply(lambda x : week_check(x))

    for d, c in zip(dt_list,class_list):
        df_tmp = data_copy[(data_copy['ClassID']==c) & (data_copy['dt']>= pd.Timestamp(d))]
        replace_idx = df_tmp.index
        class_ori = df_tmp['ClassID'].iloc[0]
        data_copy.loc[replace_idx,'ClassID']  = class_ori+'r'

    data_copy.drop(data_copy[data_copy['week'] > 10].index, inplace=True)
    data_copy.drop(data_copy[data_copy['PlantHeight'] == ''].index, inplace=True)
    data_copy.reset_index()

    data_copy['PlantHeight'] = data_copy['PlantHeight'].astype('float')
    data_copy['PlantThickness'] = data_copy['PlantThickness'].astype('float')
    data_copy['RootLength_plant'] = data_copy['RootLength_plant'].astype('int')
    data_copy['RootLength_uproot'] = data_copy['RootLength_uproot'].astype('int')
    data_copy['AirTemperature'] = data_copy['AirTemperature'].astype('float')
    data_copy['AirHumidity'] = data_copy['AirHumidity'].astype('float')
    data_copy['Co2'] = data_copy['Co2'].astype('int')
    data_copy['Quantum'] = data_copy['Quantum'].astype('int')
    data_copy['SupplyEC'] = data_copy['SupplyEC'].astype('float')
    data_copy['SupplyPH'] = data_copy['SupplyPH'].astype('int')
    data_copy['HighSoilTemp'] = data_copy['HighSoilTemp'].astype('float')
    data_copy['HighSoilHumi'] = data_copy['HighSoilHumi'].astype('float')
    data_copy['HighEC'] = data_copy['HighEC'].astype('float')
    data_copy['HighPH'] = data_copy['HighPH'].astype('float')
    data_copy['LowSoilTemp'] = data_copy['LowSoilTemp'].astype('float')
    data_copy['LowSoilHumi'] = data_copy['LowSoilHumi'].astype('float')
    data_copy['LowSoilEC'] = data_copy['LowSoilEC'].astype('float')
    data_copy['LowSoilPH'] = data_copy['LowSoilPH'].astype('float')
    data_copy['AmtIrrigation'] = data_copy['AmtIrrigation'].astype('int')
    data_copy['week'] = data_copy['week'].astype('int')

    groupby_dict = {}
    for col in feature_col_final:
        groupby_dict[col] = ['mean','min','max','std']

    data_grouped = data_copy[['ClassID','week']+feature_col_final].groupby(['ClassID','week']).agg(groupby_dict)

    c1_list = []
    c2_list = []
    for c1, c2 in data_grouped.columns:
        c1_list.append(c1)
        c2_list.append(c2)

    data_grouped.columns = [x+'_'+y for x,y in data_grouped.columns]

    for c1 in c1_list:
        data_grouped[c1+'_diff'] = data_grouped[c1+'_max'] - data_grouped[c1+'_min']

    data_grouped = data_grouped.reset_index()

    # ClassID별 최대 PlantHeight값 산출
    data_grouped_height = data_copy[['ClassID','week','PlantHeight']].groupby(['ClassID','week']).agg('max').reset_index()

    data_grouped.drop(data_grouped_height[data_grouped_height['PlantHeight'].isna()].index)
    data_grouped_height.drop(data_grouped_height[data_grouped_height['PlantHeight'].isna()].index)

    # ClassID별 최대/최소 week 
    data_grouped_week = data_grouped_height[['ClassID','week']].groupby(['ClassID']).agg(['min','max'])
    data_grouped_week.columns = [x+'_'+y for x,y in data_grouped_week.columns]
    data_grouped_week = data_grouped_week.to_dict('index')
    data_grouped.reset_index()

    data_grouped['PlantHeight'] = data_grouped_height['PlantHeight']

    return data_grouped, data_grouped_week, data_grouped_height

# 이전주차 식물 길이 생성 함수
def height_ago(row):
    if row['week'] > data_grouped_week[row['ClassID']]['week_min'] and row['week'] <= data_grouped_week[row['ClassID']]['week_max']:
        return data_grouped_height.loc[(data_grouped_height['ClassID'] == row['ClassID']) & (data_grouped_height['week'] == (row['week']-1))]['PlantHeight'].iloc[-1]
    else:
        return None

train_data_grouped, data_grouped_week, data_grouped_height = data_preprocessor(train_data, train_dt_list, train_class_list)
train_data_grouped['PlantHeight_ago'] = data_grouped_height.apply(height_ago, axis=1)

valid_data_grouped, data_grouped_week, data_grouped_height = data_preprocessor(valid_data, valid_dt_list, valid_class_list)
valid_data_grouped['PlantHeight_ago'] = data_grouped_height.apply(height_ago, axis=1)

train_data_grouped.dropna(subset=['PlantHeight', 'PlantHeight_ago'], inplace=True)
valid_data_grouped.dropna(subset=['PlantHeight', 'PlantHeight_ago'], inplace=True)

# 물주기/생육장소 컬럼 생성 함수
def Create_Char(data_grouped):
    data_grouped['locate'] = data_grouped['ClassID']
    data_grouped['water'] = data_grouped['ClassID']
    for idx, row in data_grouped.iterrows():
        if str(row['locate']).split('-')[3] == 'L':
            data_grouped.loc[idx, 'locate'] = 1
        elif str(row['locate']).split('-')[3] == 'B':
            data_grouped.loc[idx, 'locate'] = 2
        data_grouped.loc[idx, 'water'] = int(str(row['water']).split('-')[4].replace('r', ''))

    data_grouped['ClassNo'] = data_grouped['ClassID'].apply(lambda x : int(x[4:6]))
    data_grouped['ClassType'] = data_grouped['ClassID'].apply(lambda x : int(x[2:3]))
    return data_grouped

train_data_grouped = Create_Char(train_data_grouped)
valid_data_grouped = Create_Char(valid_data_grouped)


# Model Train & Valid Part
train_df = train_data_grouped.reset_index()
valid_df = valid_data_grouped.reset_index()

df_train = train_df.copy()
df_valid = valid_df.copy()

X_train = df_train.drop(['index', 'PlantHeight','ClassID'], axis=1)
X_valid = df_valid.drop(['index', 'PlantHeight','ClassID'], axis=1)

y_train = df_train['PlantHeight']
y_valid = df_valid['PlantHeight']

# 하이퍼파라미터 검색을 위한 매개변수 그리드 설정
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# GridSearchCV를 사용한 하이퍼파라미터 튜닝
rf = RandomForestRegressor(random_state=np.random.seed(int(time.time() * 1000) % 100000))
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 최적의 파라미터로 모델 훈련
best_rf = grid_search.best_estimator_

# 검증 데이터셋으로 성능 평가
y_valid_pred = best_rf.predict(X_valid)

valid_mape = mean_absolute_percentage_error(y_valid, y_valid_pred)*100
print('----------------------------------------------------------------')
print('모델 검증 MAPE :', valid_mape)

with open(file=model_path, mode='wb') as f:
    pickle.dump(best_rf, f)


print('Training 종료')
print('----------------------------------------------------------------')
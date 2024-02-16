# -*- coding: utf-8 -*-
# import libraries
import os, json, shutil, sys, argparse
import pandas as pd
import numpy as np
import pickle
import json
from tqdm import tqdm

from sklearn.metrics import mean_absolute_percentage_error, r2_score
import sklearn.exceptions

from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


OUTPUT_DIR = "/mnt/output"
DATASET_DIR =  "/mnt/dataset/생장예측/test"

# 산출물 버전관리용 파일명 접두사 설정
date_prefix = datetime.now().strftime("%Y-%m-%d")

# 산출물 결과 저장경로
result_path = os.path.join(OUTPUT_DIR, f'{date_prefix}_height_test_pred.csv')

# 모델 저장경로
model_name = f'{date_prefix}_prediction_model.abc'
model_path = os.path.join(OUTPUT_DIR, model_name)
# model_path = os.path.join(OUTPUT_DIR, f'2023-12-29_prediction_model.abc')

# 인자값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser(description="""테스트할 모델명 지정. 기본값은 오늘날짜의 모델명. 
                                 학습완료된 예제 파일명 2023-12-29_prediction_model.abc""")
# 입력받을 인자값 설정
parser.add_argument("--model_name", type=str, default=model_name)

# args 에 위의 내용 저장
args = parser.parse_args()

# 입력받은 모델명이 기본값이 아닌 경우 해당 모델명으로 모델경로 변경
if model_name!=args.model_name: model_path = os.path.join(OUTPUT_DIR, args.model_name)

# 학습된 모델 로드
if not os.path.isfile(model_path): sys.exit(f"""경로에 파일이 없습니다.{model_path}
    모델명 지정 예시 python plant_prediction_test.py --model_name 2023-12-29_prediction_model.abc""")

with open(file=model_path, mode='rb') as f:
    loaded_rf=pickle.load(f)
print(f"모델이 불러와졌습니다. {model_path}")
##########################################################################

print('시작시간', datetime.today())
# Test 데이터 셋 (json) 로드 & 통합 CSV 생성
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
for (root, directories, files) in tqdm(list(os.walk(DATASET_DIR))):
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
                        elif subCategory in ['PlantHeight',	'PlantThickness']:
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

feature_col_final = ['AirHumidity','AirTemperature','AmtIrrigation','HighEC','HighPH','HighSoilHumi','LowSoilEC',
                     'LowSoilHumi','LowSoilPH','Quantum']

# 날짜에 따른 주차 컬럼 생성
data_copy = data.copy()
data_copy['dt'] = pd.to_datetime(data_copy['GetDateTime'], format='%Y%m%d%H%M')
def week_check(d):
    week_list = [pd.Timestamp(x) for x in ["2023-08-29", "2023-09-05", "2023-09-12", "2023-09-19", "2023-09-26", "2023-10-03", "2023-10-10", "2023-10-17", "2023-10-24", "2023-10-31"]]
    for i in range(11):
        if d < week_list[i]:
            return i
        else:
            pass
    return

data_copy['week'] = data_copy['dt'].apply(lambda x : week_check(x))

data_copy.drop(data_copy[data_copy['week'] > 10].index, inplace=True)
data_copy.drop(data_copy[data_copy['PlantHeight'] == ''].index, inplace=True)
data_copy.reset_index()

# 모델 학습 활용 데이터 str -> float&int
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

# 화분/주차 컬럼 기준으로 각각의 학습 요소들에 대해 최소/최대/평균/표준편차 group by 수행
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

# 화분/주차 컬럼 기준으로 정답 라벨(PlantHeight)에 대해 group by 수행 → 특정 화분의 특정 주차에 해당하는 식물 줄기의 키
data_grouped_height = data_copy[['ClassID','week','PlantHeight']].groupby(['ClassID','week']).agg('max').reset_index()
data_grouped['PlantHeight'] = data_grouped_height['PlantHeight']

# 화분 컬럼 기준으로 주차(week)에 대해 group by 수행 → 화분별로 데이터가 존재하는 최소주차/최대주차 산출
data_grouped_week = data_grouped_height[['ClassID','week']].groupby(['ClassID']).agg(['min','max'])
data_grouped_week.columns = [x+'_'+y for x,y in data_grouped_week.columns]
data_grouped_week = data_grouped_week.to_dict('index')

# 최소주차/최대주차를 고려하여 화분/주차별로 -1주차의 식물 줄기의 키 데이터 컬럼 생성
def height_ago(row):
    if row['week'] > data_grouped_week[row['ClassID']]['week_min'] and row['week'] <= data_grouped_week[row['ClassID']]['week_max']:
        return data_grouped_height.loc[(data_grouped_height['ClassID'] == row['ClassID']) & (data_grouped_height['week'] == (row['week']-1))]['PlantHeight'].iloc[-1]
    else:
        return None
data_grouped['PlantHeight_ago'] = data_grouped_height.apply(height_ago, axis=1)
data_grouped.dropna(subset=['PlantHeight', 'PlantHeight_ago'], inplace=True)
data_grouped.reset_index()

# ClassID에 들어있는 물주기/생육장소 데이터 추출하여 컬럼 생성
data_grouped['locate'] = data_grouped['ClassID']
data_grouped['water'] = data_grouped['ClassID']
for idx, row in data_grouped.iterrows():
    if str(row['locate']).split('-')[3] == 'L':
        data_grouped.loc[idx, 'locate'] = 1
    elif str(row['locate']).split('-')[3] == 'B':
        data_grouped.loc[idx, 'locate'] = 2
    data_grouped.loc[idx, 'water'] = int(str(row['water']).split('-')[4].replace('r', ''))

df = data_grouped.reset_index()

# ClassID에 들어있는 15가지 종류의 식물/식물의 타입(건생/중생/습생/수생) 산출
df['ClassNo'] = df['ClassID'].apply(lambda x : int(x[4:6]))
df['ClassType'] = df['ClassID'].apply(lambda x : int(x[2:3]))
df_test = df.copy()
print('shape of test:',df_test.shape, '-- 고유 ClassID 개수:', df_test['ClassID'].nunique())

# X값 제외 drop
X_test = df_test.drop(['index','PlantHeight','ClassID'], axis=1)
# Y값 라벨 설정
y_test = df_test['PlantHeight']

# 테스트셋을 통한 예측값 산출
y_test_pred = loaded_rf.predict(X_test)

# ClassID의 특정 상태값 별 실제 줄기값과 모델이 예측한 줄기값 CSV 추출
df_pred = df[['ClassID', 'PlantHeight']].copy()
df_pred['Preds'] = y_test_pred
print('df_pred:', df_pred)
df_pred.to_csv(result_path, sep=',')
print("산출물 저장경로", result_path)

# MAPE 평가지표 산출
mape = mean_absolute_percentage_error(y_test, y_test_pred)*100

print('----------------------------------------------------------------')
print('TEST MAPE : ', mape)
print('----------------------------------------------------------------')

print('종료시간', datetime.today())
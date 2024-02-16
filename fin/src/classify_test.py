# -*- coding: utf-8 -*-
# 2023-12-07 참조 원본파일명
# 이미지분류모델_v5.041_crop_and_test.py
import os, json, shutil, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, MaxPooling2D, concatenate,Input, Flatten
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import Xception

from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

## 파라미터 설정 ##########################################################
OUTPUT_DIR   = "/mnt/output" # 혼동 행렬, 전처리된 데이터셋 저장경로(쓰기권한필요)
DATASET_DIR  = "/mnt/dataset/이미지분류/test" # 데이터셋 경로(읽기권한필요)
NUM_OF_BATCH = 32
INPUT_SIZE   = (128, 128) # crop된 이미지의 resize 크기. 모델의 조건 (train 당시의 세팅 값)과 맞춰야 함.
##########################################################################

# 산출물 버전관리용 파일명 접두사 설정
date_prefix = datetime.now().strftime("%Y-%m-%d")

# 개별결괏값 및 혼동행렬 저장경로 지정
result_path = os.path.join(OUTPUT_DIR, f"{date_prefix}_img_classification_test_pred.csv")
excel_path = os.path.join(OUTPUT_DIR, f'{date_prefix}_confusion_matrix.xlsx')
heatmap_path = os.path.join(OUTPUT_DIR, f'{date_prefix}_confusion_matrix.png')

# 모델경로 지정
model_name = f"{date_prefix}_img_model.h5"
model_path = os.path.join(OUTPUT_DIR, model_name)
# model_path = os.path.join(OUTPUT_DIR, 'img_model_2024-01-09.h5')

# 인자값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser(description="""테스트할 모델명 지정. 기본값은 오늘날짜의 모델명. 
                                 학습완료된 예제 파일명 img_model_2024-01-09.h5""")
# 입력받을 인자값 설정
parser.add_argument("--model_name", type=str, default=model_name)

# args 에 위의 내용 저장
args = parser.parse_args()

# 입력받은 모델명이 기본값이 아닌 경우 해당 모델명으로 모델경로 변경
if model_name!=args.model_name: model_path = os.path.join(OUTPUT_DIR, args.model_name)

# 학습된 모델 로드
if not os.path.isfile(model_path): sys.exit(f"""경로에 파일이 없습니다.{model_path}
    모델명 입력 예시 python classify_test.py --model_name img_model_2024-01-09.h5""")
model = load_model(model_path)
print(f"모델이 불러와졌습니다.{model_path}")


## 데이터 로드할 폴더명 지정
jpg_folder = os.path.join(DATASET_DIR, "이미지데이터")
json_folder = os.path.join(DATASET_DIR, "라벨링데이터")

# 전처리 파일 저장경로
# cropped_path = os.path.join(OUTPUT_DIR, "이미지데이터_cropped") # 자른 테스트 이미지 경로
cropped_path = os.path.join(OUTPUT_DIR, "img_cropped_test") # 자른 train 이미지 경로
if not os.path.isdir(cropped_path):
    os.makedirs(cropped_path)
else:
    try: shutil.rmtree(cropped_path)
    except Exception as err: print(f"can't clean cropped path: {cropped_path}")
    os.makedirs(cropped_path)

# 모델 input값 확인용 image 저장 경로
input_img_path = ''
# input_img_path = os.path.join(OUTPUT_DIR, "이미지데이터_input") # null일 경우, 저장 안함
# if not os.path.isdir(input_img_path): os.makedirs(input_img_path)

# 한글 폰트 설정
# font_path = 'C:/Windows/Fonts/malgun.ttf'  # 윈도우 맑은 고딕 폰트의 경로
font_path = '/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf' # 컨테이너의 경우 폰트의 경로
##########################################################################

# 파일 경로 수집 함수
def get_file_paths(folder, extension):
    paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(extension):
                paths.append(os.path.join(root, file))
    return paths

# 파일 경로 수집 함수 (딕셔너리 형태로 생성)
def get_file_map_paths(folder, extension):
    paths = {}
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(extension):
                f = os.path.join(root, file)
                paths[os.path.basename(f).split('.')[0]] = f
    return paths

# JSON 파일에서 라벨과 폴리곤 정보 추출 함수
def extract_label_and_polygon(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    plant_name = data["plant"]["PlantName"]
    polygon = data["annotations"]
    pic = [int(x) for x in data["picInfo"]["ImageResolution"].split(' * ')]
    return plant_name, polygon, pic


# 시작 시간 기록
start_time = datetime.now()

print(f'\nscan JPEG dir in "{jpg_folder}" ...')
image_paths = get_file_paths(jpg_folder, ".jpg")
print(f'\nscan JSON dir "{json_folder}" ...')
all_json_paths = get_file_map_paths(json_folder, ".json")
print(f'\n{len(image_paths)} images and {len(all_json_paths)} JSON files scaned')

# 이미지 파일의 확장자 제거한 파일 이름 목록을 만듬
file_names = [os.path.basename(f).split('.jpg')[0] for f in image_paths]

# 모든 라벨과 폴리곤 정보 추출
json_paths = [all_json_paths[fname] for fname in file_names if all_json_paths.get(fname,0)]
print(f'\nextracting label and polygon {len(image_paths)} out of {len(image_paths)}...')
labels_and_polygons = [extract_label_and_polygon(fpath) for fpath in json_paths]
labels, polygons, pics = zip(*labels_and_polygons)  # 리스트를 라벨과 폴리곤 리스트로 분리

# 라벨 인코딩 및 원-핫 인코딩
label_encoder = LabelEncoder()
all_labels_tuple = [('A-1-01','스투키'),
                   ('A-1-02','선인장'),
                   ('A-1-03','금전수'),
                   ('A-2-04','테이블야자'),
                   ('A-2-05','홍콩야자'),
                   ('A-2-06','호접란'),
                   ('A-2-07','스파티필럼'),
                   ('A-3-08','보스턴고사리'),
                   ('A-3-09','몬스테라'),
                   ('A-4-10','부레옥잠'),
                   ('B-1-11','올리브나무'),
                   ('B-2-12','오렌지쟈스민'),
                   ('B-2-13','관음죽'),
                   ('B-3-14','벵갈고무나무'),
                   ('B-3-15','디펜바키아')]
id_to_name_dict = {k[0]:k[1] for k in all_labels_tuple}
name_to_id_dict = {k[1]:k[0] for k in all_labels_tuple}
label_encoder.fit(list(id_to_name_dict.keys()))
labels_encoded = label_encoder.transform([name_to_id_dict[x] for x in labels])
labels_categorical = to_categorical(labels_encoded)
    
## 폴리곤 정보에 맞춰서 이미지 잘라서 저장 ##########################################################
def crop_polygon_image(image_path, polygon_coords):
    """
    이미지에서 폴리곤 영역을 추출하고 해당 영역을 크롭합니다.

    :param image_path: 이미지 파일 경로
    :param polygon_coords: 폴리곤 좌표 리스트 [(x1, y1), (x2, y2), ...]
    :return: 크롭된 이미지 객체
    """
    # 이미지 로드
    image = Image.open(image_path)

    # 폴리곤 마스크 생성
    mask = Image.new('L', image.size, 0)
    ImageDraw.Draw(mask).polygon(polygon_coords, outline=1, fill=1)
    mask = np.array(mask)

    # 원본 이미지에 마스크 적용
    masked_image = np.array(image) * np.expand_dims(mask, axis=-1)

    # 바운딩 박스 계산
    nonzero_coords = np.argwhere(mask)
    top_left = nonzero_coords.min(axis=0)
    bottom_right = nonzero_coords.max(axis=0)
    cropped_image = masked_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    # PIL 이미지로 변환
    cropped_image = Image.fromarray(cropped_image)
    return cropped_image

def crop_center(image_path, new_width=600, new_height=800):
    image = Image.open(image_path)
    width, height = image.size
    left = (width - new_width)/2
    top = (height - new_height)/3
    right = (width + new_width)/2
    bottom = (height + new_height)/3
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

def crop_image_job(img_path, polygons, outfile_prefix, json_path):
    cnt = 0
    for j in range(len(polygons)-1, -1, -1):
        polygon = polygons[-1]
        x = polygon['plant_polygon']
        # [(x1, y1), (x2, y2), ...] 형식 좌표로 만듬
        polygon_coordinates = [(x[i], x[i + 1]) for i in range(0, len(x), 2)]
        new_filename = f'{outfile_prefix}.jpg'
        dst_img_path = os.path.join(cropped_path, new_filename)
        try:
            cropped_img = crop_polygon_image(img_path, polygon_coordinates)
            cropped_img.save(dst_img_path)
            cnt += 1
            break
        except Exception as e:
            # 예외가 발생한 경우 처리
            # print(f'Exception occurred for image: {img_path}')
            #print(f'Error: {e}')
            if j == 0:
                cropped_img = crop_center(img_path, 700, 800)
                #print(f'Save crop', dst_img_path)
                cropped_img.save(dst_img_path)
                cnt += 1
    return cnt

def process_images(args):
    img_path, polygons, outfile_prefix, json_path = args
    return crop_image_job(img_path, polygons, outfile_prefix, json_path)


tasks = [(img_path, labels_and_polygons[i][1], file_names[i], json_paths[i]) for i, img_path in enumerate(image_paths)]
tasks_df = pd.DataFrame(tasks)
tasks_df['f_name'] = tasks_df[0].apply(lambda x : os.path.basename(x))
tasks_done_df = pd.DataFrame({os.path.basename(f).replace('_0.jpg','.jpg').replace('_1.jpg','.jpg') for f in get_file_paths(cropped_path, 'jpg')},columns=['f_name'])
drop_idx = pd.merge(tasks_df, tasks_done_df, on='f_name',how='inner').index
tasks_to_do_df = tasks_df.loc[[x for x in tasks_df.index if x not in drop_idx]].drop('f_name',axis=1)
tasks_to_do = []
for i in tasks_to_do_df.index:
    tasks_to_do.append(tasks_to_do_df.loc[i].to_list())
print(f'\nOut of a total of {len(image_paths)} images, we found {len(image_paths)-len(tasks_to_do)} already cropped images, and we will perform the cropping process on {len(tasks_to_do)} images.')
print(f'\nrun {len(tasks_to_do)} tasks ...')
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_images, task) for task in tasks_to_do]
    print(f'with {len(futures)} futures ...')
    for future in tqdm(as_completed(futures), total=len(futures)):
        _ = future.result()
print(f"Total processed images: {sum(f.result() for f in futures)}")
    
## 폴리곤 형태로 잘린 이미지 경로 로드 ##########################################################
croped_image_paths = get_file_paths(cropped_path, ".jpg")

# 모든 라벨과 폴리곤 정보 추출
labels_test = [os.path.basename(json_path)[4:10] for json_path in croped_image_paths]

# 라벨 인코딩 및 원-핫 인코딩
labels_test_encoded = label_encoder.transform(labels_test)
labels_test_categorical = to_categorical(labels_test_encoded)
classes_list = [str(x) for x in label_encoder.transform(label_encoder.classes_)]

# 이미지 경로와 라벨, 폴리곤을 DataFrame으로 생성
test_df = pd.DataFrame({
    'image_path': croped_image_paths,
    'label': labels_test_encoded,
})

test_df['class_name'] = test_df['image_path'].apply(lambda x : os.path.basename(x)[4:14])

# 각 class_name 개수와, label 개수 출력
print('테스트용 데이터 셋 개수:', test_df.shape[0])
print('테스트용 데이터 셋의 고유 조건의 개수:', test_df['class_name'].nunique(), test_df['class_name'].unique())
print('테스트용 데이터 셋의 고유 class (label)의 개수:', len(set(labels_test)), [id_to_name_dict[x] for x in label_encoder.classes_ if x in set(labels_test)])

# 문자열로 된 라벨 컬럼 생성.
test_df['label_str'] = test_df['label'].apply(lambda x: str(x))

# 검증 및 테스트 데이터에 적용할 ImageDataGenerator (증강 없음)
test_datagen = ImageDataGenerator(rescale=1./255,       
    )

# Test dataset generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_path',
    y_col='label_str',
    classes=classes_list,
    target_size=INPUT_SIZE,
    batch_size=NUM_OF_BATCH,
    class_mode='categorical',
    shuffle=False , # 순서를 유지합니다.              
    save_to_dir= os.path.join(input_img_path,'img_test') if input_img_path else None,  # 증강된 이미지를 저장할 경로
    save_format='jpeg' if input_img_path else None  # 저장할 이미지 형식
)


# 소요 시간 출력
print(f"총 소요된 시간: {datetime.now()-start_time}")
print("### 데이터 준비완료 ######################################################")

### 학습완료된 모델 로드 및 성능검증 ####################################################
start_time = datetime.now()
# Perform predictions on the test dataset
y_pred = model.predict(test_generator, steps=len(test_generator))
y_pred_classes = np.argmax(y_pred, axis=1)

# 테스트 생성기에서 실제 레이블 가져오기
y_true = test_generator.classes

# 길이가 일치하는지 확인합니다
print(f"Length of y_true: {len(y_true)}")
print(f"Length of y_pred_classes: {len(y_pred_classes)}")

# classification_report 전 길이가 일치하는지 확인합니다
if len(y_true) == len(y_pred_classes):
    # 숫자 레이블을 원래 문자 레이블로 변환합니다.
    y_true_labels_d = [id_to_name_dict[x] for x in label_encoder.inverse_transform(y_true)]
    y_pred_labels_d = [id_to_name_dict[x] for x in label_encoder.inverse_transform(y_pred_classes)]
    target_names = [x for x in id_to_name_dict.values() if x in y_true_labels_d]
    # 원래 문자 레이블을 사용하여 classification_report를 생성합니다.
    report = classification_report(y_true_labels_d, y_pred_labels_d, labels = target_names, target_names = target_names)
    print(report)
else:
    print("Length of y_true and y_pred_classes do not match.")

# # 중간산출물 제거
# try: shutil.rmtree(cropped_path)
# except Exception as err: print(f"Failed to delete!!!!! {err}")

# 소요 시간 출력
print(f"총 소요된 시간: {datetime.now()-start_time}")


#### 개별결괏값 및 혼동행렬 저장 ############################################
test_df = test_df[['image_path']]
test_df['image_path'] = test_df['image_path'].apply(lambda x : os.path.basename(x).replace('_0.jpg','.jpg').replace('_1.jpg','.jpg'))
test_df['ClassName'] = y_true_labels_d
test_df['preds'] = y_pred_labels_d
test_df.to_csv(result_path)

# 혼동 행렬 계산
conf_class_list = [id_to_name_dict[x] for x in label_encoder.classes_]
conf_matrix = confusion_matrix(y_true_labels_d, y_pred_labels_d, labels=conf_class_list)

# 혼동 행렬을 DataFrame으로 변환
conf_matrix_df = pd.DataFrame(conf_matrix, index=conf_class_list, columns=conf_class_list)
conf_matrix_df.to_excel(os.path.join(OUTPUT_DIR, "conf_mat_df.xlsx"))

# 각 클래스에 대한 Total, Recall, Precision, F1-score 계산
total = conf_matrix_df.sum(axis=1)
recall = np.diag(conf_matrix_df) / np.sum(conf_matrix_df, axis=1)
recall = recall.replace([np.inf, np.nan], 0)
precision = np.diag(conf_matrix_df) / np.sum(conf_matrix_df, axis=0)
precision = precision.replace([np.inf, np.nan], 0)
f1 = 2 * (precision * recall) / (precision + recall)

# 0으로 된 값 채우기
recall = recall.fillna(0)
precision = precision.fillna(0)
f1 = f1.fillna(0)

# 계산된 값들을 DataFrame에 추가
conf_matrix_df['Total'] = total
conf_matrix_df['Recall'] = recall
conf_matrix_df['Precision'] = precision
conf_matrix_df['F1-score'] = f1

# 전체 평균 (mean) 추가
conf_matrix_df.loc['Mean'] = conf_matrix_df[conf_matrix_df['Total']>0].mean()

# 엑셀 파일로 저장
conf_matrix_df.to_excel(excel_path)
print(f"혼동 행렬이 '{excel_path}' 파일로 저장되었습니다.")

# 시각화
try:
    # 한글 폰트 설정
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)

    fig, ax = plt.subplots(figsize=(10, 10))  
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap='Blues', 
                xticklabels=conf_class_list,
                yticklabels=conf_class_list)

    # 라벨 설정
    ax.set_xlabel('예측 레이블')
    ax.set_ylabel('실제 레이블')
    ax.set_title('혼동 행렬')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # plt.show()
    plt.savefig(heatmap_path)
    print(f"혼동 행렬이 '{heatmap_path}' 파일로 저장되었습니다.")
except Exception as e:
    print(e)
    pass

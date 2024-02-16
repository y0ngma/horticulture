import os, sys, json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, MaxPooling2D, concatenate,Input, Flatten
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import Xception

from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# 병렬로 JSON 정보 추출
def extract_labels_and_polygons_parallel(json_paths):
    with ThreadPoolExecutor() as executor:
        return list(executor.map(extract_label_and_polygon, json_paths))

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
    if nonzero_coords.size > 0: # Fix error: ValueError: zero-size array to reduction operation minimum which has no identity
        top_left = nonzero_coords.min(axis=0)
        bottom_right = nonzero_coords.max(axis=0)
    else:
        # print(f"can't get masked image: {image_path}")
        top_left = (0, 0)   ## mask이미지를 얻지 못할때 5x5 이미지 생성
        bottom_right = (5, 5)
    cropped_image = masked_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    # PIL 이미지로 변환
    cropped_image = Image.fromarray(cropped_image)
    return cropped_image

def crop_image_job(img_path, polygons, outfile_prefix, json_path, cropped_path):
    cnt = 0
    for j, polygon in enumerate(polygons):
        x = polygon['plant_polygon']
        # [(x1, y1), (x2, y2), ...] 형식 좌표로 만듬
        polygon_coordinates = [(x[i], x[i + 1]) for i in range(0, len(x), 2)]
        new_filename = f'{outfile_prefix}_{j}.jpg'
        if len(polygon_coordinates) >= 3:
            try:
                cropped_img = crop_polygon_image(img_path, polygon_coordinates)
                if cropped_img.size[0] >= 80 and cropped_img.size[1] >= 80:
                    new_filename = f'{outfile_prefix}_{j}.jpg'
                    dst_img_path = os.path.join(cropped_path, new_filename)
                    cropped_img.save(dst_img_path)
                    cnt += 1
                    #print(f'[{save_cnt}] {img_path} -> {dst_img_path}')
                else:
                    pass
                    #print(f'too small image ignored: {img_path}')
            except Exception as e:
                # 예외가 발생한 경우 처리
                print(f'Exception occurred for image: {img_path}')
                print(f'Error: {e}')
        else:
            # print(f'invalid polygon coords: {json_path} {x}')
            pass
    return cnt

def process_images(args):
    img_path, polygons, outfile_prefix, json_path, cropped_path = args
    return crop_image_job(img_path, polygons, outfile_prefix, json_path, cropped_path)

# 사용자 정의 콜백.
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            if logs.get('accuracy') > 0.98:  
                print("\n정확도가 95%를 넘었으므로 학습을 중단합니다!")
                self.model.stop_training = True

class CustomMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, interval=50):
        super(CustomMetricsCallback, self).__init__()
        self.validation_data = validation_data  # 검증 데이터셋
        self.interval = interval  # 간격 설정
        self.val_iterator = iter(validation_data)  # 검증 데이터셋 이터레이터 생성

    def on_batch_end(self, batch, logs=None):
        if batch % self.interval == 0:
            try:
                x_val, y_val = next(self.val_iterator)
            except StopIteration:
                # 검증 데이터셋의 끝에 도달하면, 이터레이터를 다시 시작합니다.
                self.val_iterator = iter(self.validation_data)
                x_val, y_val = next(self.val_iterator)

            predictions = self.model.predict(x_val, verbose=0)
            # 원-핫 인코딩된 라벨을 정수 인코딩으로 변환
            y_val = np.argmax(y_val, axis=1)
            predictions = np.argmax(predictions, axis=1)

            acc = accuracy_score(y_val, predictions)
            f1 = f1_score(y_val, predictions, average='weighted')

            print(f" Iteration {batch}: acc = {acc:.4f}, F1 = {f1:.4f}")

if __name__ == "__main__":
    ## 파라미터 설정 ##########################################################
    OUTPUT_DIR   = "/mnt/output" # 학습된 모델 파일과 혼동 행렬 저장 경로
    DATASET_DIR  = "/mnt/dataset/이미지분류/" #  경로(자르기 전)

    NUM_OF_BATCH = 32
    NUM_OF_EPOCH = 1
    INPUT_SIZE   = (128, 128) # crop된 이미지의 resize 크기
    FILTERS      = 16
    ##########################################################################

    # 데이터 로드할 폴더명 지정
    jpg_folder_train = os.path.join(DATASET_DIR, "train", "이미지데이터")
    json_folder_train = os.path.join(DATASET_DIR, "train", "라벨링데이터")
    jpg_folder_val = os.path.join(DATASET_DIR, "val", "이미지데이터")
    json_folder_val = os.path.join(DATASET_DIR, "val", "라벨링데이터")

    # 전처리 파일 저장경로
    cropped_path = os.path.join(OUTPUT_DIR, "img_cropped") # 자른 train 이미지 경로
    if not os.path.isdir(cropped_path): os.makedirs(cropped_path)

    # 모델 input값 확인용 image 저장 경로
    input_img_path = ''
    # input_img_path = os.path.join(OUTPUT_DIR, "이미지데이터_input") # null일 경우, 저장 안함
    # if not os.path.isdir(input_img_path): os.makedirs(input_img_path)

    # 산출물 버전관리용 파일명 접두사 설정
    date_prefix = pd.Timestamp.today().strftime('%Y-%m-%d')

    # 모델을 저장할 경로와 파일 이름을 지정합니다.
    saved_model_path = os.path.join(OUTPUT_DIR, f"{date_prefix}_img_model.h5")
    print('모델을 저장할 경로및 파일 이름',saved_model_path)

    ##########################################################################
    # 시작 시간 기록
    start_time = datetime.now()
    print(datetime.now() - start_time, 'Start.')

    print(f'\nscan JPEG dir in "{jpg_folder_train} and {jpg_folder_val}" ...')
    image_paths_train = get_file_paths(jpg_folder_train, ".jpg")
    image_paths_val = get_file_paths(jpg_folder_val, ".jpg")
    image_paths = image_paths_train + image_paths_val
    print(f'\nscan JSON dir "{json_folder_train} and {json_folder_val}" ...')
    all_json_paths_train = get_file_map_paths(json_folder_train, ".json")
    all_json_paths_val = get_file_map_paths(json_folder_val, ".json")

    print(f'\n{len(image_paths_train)} images and {len(all_json_paths_train)} JSON files scaned for training')
    print(f'\n{len(image_paths_val)} images and {len(all_json_paths_val)} JSON files scaned for validating')
    print(datetime.now() - start_time, 'Finised scanning files.')

    # 이미지 파일의 확장자 제거한 파일 이름 목록을 만듬
    file_names_train = [os.path.basename(f).split('.jpg')[0] for f in image_paths_train]
    file_names_val = [os.path.basename(f).split('.jpg')[0] for f in image_paths_val]
    file_names = [os.path.basename(f).split('.jpg')[0] for f in image_paths]

    # 모든 라벨과 폴리곤 정보 추출
    json_paths_train = [all_json_paths_train[fname] for fname in file_names_train if all_json_paths_train.get(fname,0)]
    json_paths_val = [all_json_paths_val[fname] for fname in file_names_val if all_json_paths_val.get(fname,0)]
    json_paths = json_paths_train + json_paths_val
    print(f'\nextracting label and polygon {len(image_paths)} out of {len(image_paths)}...')
    labels_and_polygons = extract_labels_and_polygons_parallel(json_paths)
    labels, polygons, pics = zip(*labels_and_polygons)  # 리스트를 라벨과 폴리곤 리스트로 분리
    print(datetime.now() - start_time, 'Finised extracting label and polygon.')

    # 라벨 인코딩 및 원-핫 인코딩
    all_labels_tuple = [('N50-A-1-01','스투키'),
                        ('N50-A-1-02','선인장'),
                        ('N50-A-1-03','금전수'),
                        ('N50-A-2-04','테이블야자'),
                        ('N50-A-2-05','홍콩야자'),
                        ('N50-A-2-06','호접란'),
                        ('N50-A-2-07','스파티필럼'),
                        ('N50-A-3-08','보스턴고사리'),
                        ('N50-A-3-09','몬스테라'),
                        ('N50-A-4-10','부레옥잠'),
                        ('N50-B-1-11','올리브나무'),
                        ('N50-B-2-12','오렌지쟈스민'),
                        ('N50-B-2-13','관음죽'),
                        ('N50-B-3-14','벵갈고무나무'),
                        ('N50-B-3-15','디펜바키아')]
    label_encoder = LabelEncoder()
    id_to_name_dict = {k[0]:k[1] for k in all_labels_tuple}
    name_to_id_dict = {k[1]:k[0] for k in all_labels_tuple}
    label_encoder.fit(list(id_to_name_dict.keys()))
    labels_encoded = label_encoder.transform([name_to_id_dict[x] for x in labels])
    labels_categorical = to_categorical(labels_encoded)
        
    ## 폴리곤 정보에 맞춰서 이미지 잘라서 저장 ##########################################################
    tasks = [(img_path, labels_and_polygons[i][1], file_names[i], json_paths[i]) for i, img_path in enumerate(image_paths)]
    tasks_df = pd.DataFrame(tasks)
    tasks_df['f_name'] = tasks_df[0].apply(lambda x : os.path.basename(x))
    tasks_done_df = pd.DataFrame({os.path.basename(f).replace('_0.jpg','.jpg').replace('_1.jpg','.jpg') for f in get_file_paths(cropped_path, 'jpg')},columns=['f_name'])
    drop_idx = pd.merge(tasks_df, tasks_done_df, on='f_name',how='inner').index
    tasks_to_do_df = tasks_df.loc[[x for x in tasks_df.index if x not in drop_idx]].drop('f_name',axis=1)
    tasks_to_do_df['cropped_path'] = cropped_path
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
    print(datetime.now() - start_time, 'Finised cropping images.')
        
    ## 폴리곤 형태로 잘린 이미지 경로 로드 ##########################################################
    croped_image_paths = get_file_paths(cropped_path, ".jpg")

    # 모든 라벨과 폴리곤 정보 추출
    labels_train = [os.path.basename(json_path)[:10] for json_path in croped_image_paths]

    # 라벨 인코딩 및 원-핫 인코딩
    labels_train_encoded = label_encoder.transform(labels_train)
    labels_train_categorical = to_categorical(labels_train_encoded)
    classes_list = [str(x) for x in label_encoder.transform(label_encoder.classes_)]

    # 이미지 경로와 라벨, 폴리곤을 DataFrame으로 생성
    df = pd.DataFrame({
        'image_path': croped_image_paths,
        'label': labels_train_encoded,
    })
    df['raw_file_name'] = df['image_path'].apply(lambda x : os.path.basename(x).replace('_0.jpg','.jpg').replace('_1.jpg','.jpg'))
    train_df = df[df['raw_file_name'].isin([os.path.basename(path) for path in image_paths_train])]
    valid_df = df[df['raw_file_name'].isin([os.path.basename(path) for path in image_paths_val])]

    # 문자열로 된 라벨 컬럼 생성.
    train_df['label_str'] = train_df['label'].apply(lambda x: str(x))
    valid_df['label_str'] = valid_df['label'].apply(lambda x: str(x))

    # 각 class_name 개수와, label 개수 출력
    print('학습용 데이터 셋 개수:', train_df.shape[0])
    print('학습용 데이터 셋의 고유 class (label)의 개수:', len(set(labels_train)), [id_to_name_dict[x] for x in label_encoder.classes_ if x in set(labels_train)])

    
    df.to_csv(os.path.join(OUTPUT_DIR,'df_trva_final.csv'),encoding='euc-kr')
    train_df.to_csv(os.path.join(OUTPUT_DIR,'train_df_final.csv'),encoding='euc-kr')
    valid_df.to_csv(os.path.join(OUTPUT_DIR,'valid_df_final.csv'),encoding='euc-kr')

    # 이미지 데이터 제너레이터 생성
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # 픽셀 값을 0과 1 사이로 정규화
        rotation_range=90,  # 무작위 회전 (0-20도)
        width_shift_range=0.2,  # 수평 이동
        height_shift_range=0.2,  # 수직 이동
        shear_range=0.2,  # 전단 변환
        zoom_range=0.2,  # 무작위 줌
        horizontal_flip=True,  # 수평 뒤집기
        fill_mode='nearest',  # 새로 생성된 픽셀 채우기 방식
    )

    # 훈련 데이터셋 제너레이터
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='label_str', 
        classes=classes_list, 
        target_size=INPUT_SIZE,
        batch_size=NUM_OF_BATCH,
        class_mode='categorical',
        # shuffle=False  # 순서를 유지합니다.
        save_to_dir= os.path.join(input_img_path,'img_train') if input_img_path else None,  # 증강된 이미지를 저장할 경로
        save_format='jpeg' if input_img_path else None  # 저장할 이미지 형식
    )

    # 검증 데이터셋 제너레이터
    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col='image_path',
        y_col='label_str',  
        classes=classes_list,
        target_size=INPUT_SIZE,
        batch_size=NUM_OF_BATCH,
        class_mode='categorical',
        # shuffle=False  # 순서를 유지합니다.
        save_to_dir= os.path.join(input_img_path,'img_valid') if input_img_path else None,  # 증강된 이미지를 저장할 경로
        save_format='jpeg' if input_img_path else None  # 저장할 이미지 형식
    )



    # 첫 번째 배치의 데이터와 라벨을 가져옵니다.
    x_val, y_val = next(iter(validation_generator))

    # 데이터 배치의 크기를 확인합니다.
    print(f"Data batch shape: {x_val.shape}")
    print(f"Labels batch shape: {y_val.shape}")

    # 소요 시간 출력
    end_time = datetime.now()
    print(f"총 소요된 시간: {end_time - start_time}")

    print("### 데이터 준비완료 ######################################################")


    ## 모델 정의 및 학습 ##########################################################
    start_time = datetime.now()

    # Inception 모듈 정의
    input_tensor = Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
    x0 = input_tensor

    # Inception 브랜치 정의
    branch_a = Conv2D(FILTERS, 1, activation='relu', padding='same')(input_tensor)
    branch_b = Conv2D(FILTERS, 1, activation='relu', padding='same')(input_tensor)
    branch_b = Conv2D(FILTERS, 3, activation='relu', padding='same')(branch_b)
    branch_c = Conv2D(FILTERS, 1, activation='relu', padding='same')(input_tensor)
    branch_c = Conv2D(FILTERS, 5, activation='relu', padding='same')(branch_c)
    branch_d = MaxPooling2D(3, strides=1, padding='same')(input_tensor)
    branch_d = Conv2D(FILTERS, 1, activation='relu', padding='same')(branch_d)
    x1 = concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

    # Xception 모델
    base_model = Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)
    base_model.trainable = True
    for layer in base_model.layers:
        if "batch_normalization" in layer.name:
            layer.trainable = False
    x2 = base_model.output

    # GlobalAveragePooling2D 적용
    x0_pool = GlobalAveragePooling2D()(x0)
    x1_pool = GlobalAveragePooling2D()(x1)
    x2_pool = GlobalAveragePooling2D()(x2)

    # 병합하여 PolyNet Model 모델 구축
    x_merged = concatenate([x0_pool, x1_pool, x2_pool], axis=-1)
    x_merged = Dense(1024, activation='relu')(x_merged)
    x_merged = Dropout(0.4)(x_merged)

    # 라벨의 클래스 수 계산
    num_classes = np.max(labels_encoded) + 1

    # 마지막 층 정의
    predictions = Dense(num_classes, activation='softmax')(x_merged)

    # 최종 모델 정의
    model = Model(inputs=input_tensor, outputs=predictions)

    # 모델 컴파일 및 요약
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #print(model.summary())

    # 파라미터 설정
    batch_size = NUM_OF_BATCH
    steps_per_epoch = len(train_df) // batch_size

    # 검증 스텝 수 계산
    validation_steps = len(validation_generator) // batch_size

    ## 모델 학습
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)


    # 커스텀 콜백 인스턴스 생성
    custom_metrics_callback = CustomMetricsCallback(validation_generator, interval=50)

    # 콜백 리스트에 CustomCallback 인스턴스 추가
    custom_callback = CustomCallback()

    # 학습 시작
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=NUM_OF_EPOCH,
        validation_data=validation_generator,
        validation_steps=validation_steps, # validation_steps도 정의해야 함
        callbacks=[early_stopping, custom_callback, custom_metrics_callback],
        verbose=1
    )

    # 모델 저장
    model.save(saved_model_path)
    print(f'모델이 저장되었습니다. 모델 파일명:{saved_model_path}')

    # 소요 시간 출력
    print(f"총 소요된 시간: {datetime.now()-start_time}")
    print("## 학습 완료 ##########################################################")

    # # 중간산출물 제거()
    # print("Deleting cropped img in...", cropped_path)
    # try: shutil.rmtree(cropped_path)
    # except Exception as err: print(f"Failed to delete!!!!! {err}")
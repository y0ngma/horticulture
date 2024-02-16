import os, sys, json, shutil
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


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


def copy_file_to_directory(file_path, target_directory):
    """
    파일을 주어진 디렉토리로 복사합니다. 
    대상 디렉토리가 존재하지 않으면 생성합니다.

    :param file_path: 복사할 파일의 경로
    :param target_directory: 파일을 복사할 대상 디렉토리
    """
    # 파일 존재 여부 확인
    if not os.path.isfile(file_path):
        print(f"Error: '{file_path}' does not exist or is not a file.")
        return

    # 대상 디렉토리 존재 여부 확인, 없으면 생성
    if not os.path.isdir(target_directory):
        # print(f"'{target_directory}' does not exist. Creating directory.")
        os.makedirs(target_directory, exist_ok=True)

    # 파일 이름 추출 (경로 제외)
    file_name = os.path.basename(file_path)

    # 대상 경로 생성
    target_path = os.path.join(target_directory, file_name)

    # 파일 복사
    shutil.copy(file_path, target_path)


def copy_files_in_parallel(file_paths, target_directories):
    with ThreadPoolExecutor() as executor:
        executor.map(copy_file_to_directory, file_paths, target_directories)


if __name__ == "__main__":
    ## 파라미터 설정 ##########################################################
    OUTPUT_DIR   = "/home/yh-jung/home/output" # 저장 경로
    DATASET_DIR  = "/home/gocp/mnt/final/이미지분류/all" # 원본 전체데이터셋 경로(자르기 전)
    
    RANDOM_SEED  = 45
    SPLIT_RATIO  = (8,1,1)
    ##########################################################################
    # 데이터 로드할 폴더명 지정
    jpg_folder = os.path.join(DATASET_DIR, "이미지데이터")
    json_folder = os.path.join(DATASET_DIR, "라벨링데이터")

    # 시작 시간 기록
    start_time = datetime.now()

    # 산출물 버전관리용 파일명 접두사 설정
    date_prefix = datetime.now().strftime("%Y-%m-%d")

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
    labels = [os.path.basename(fpath)[:10] for fpath in json_paths]
    print(labels[:20])

    # 라벨 인코딩 및 원-핫 인코딩
    label_encoder = LabelEncoder()
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
    id_to_name_dict = {k[0]:k[1] for k in all_labels_tuple}
    name_to_id_dict = {k[1]:k[0] for k in all_labels_tuple}
    label_encoder.fit(list(id_to_name_dict.keys()))
    labels_encoded = label_encoder.transform(labels)
    labels_categorical = to_categorical(labels_encoded)
        
    df = pd.DataFrame({
        'image_path': image_paths,
        'json_paths': json_paths,
        'label': labels_encoded,
    })

    df.to_csv(os.path.join(OUTPUT_DIR, f'{date_prefix}_all_df.csv'),encoding='euc-kr')
    print(df.head())
    print(df.shape)
    print(df.info())

    df = pd.read_csv(os.path.join(OUTPUT_DIR, f'{date_prefix}_all_df.csv'), encoding='euc-kr', index_col=0)

    train_df_list = []
    valid_df_list = []
    test_df_list = []

    for label in sorted(df['label'].unique()):
        print(label)
        df_tmp = df[df['label']==label]

        # 데이터를 Train/Valid/Test 로 나누기
        train_df_tmp, temp_df = train_test_split(df_tmp, test_size=0.2, random_state=RANDOM_SEED)
        test_df_tmp, valid_df_tmp = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED)

        train_df_list.append(train_df_tmp)
        valid_df_list.append(valid_df_tmp)
        test_df_list.append(test_df_tmp)

    train_df = pd.concat(train_df_list)
    valid_df = pd.concat(valid_df_list)
    test_df = pd.concat(test_df_list)

    print(train_df.groupby(['label']).count())
    print(valid_df.groupby(['label']).count())
    print(test_df.groupby(['label']).count())

    train_df['td_jpg_path'] = train_df['image_path'].apply(lambda x : os.path.dirname(x).replace('/all','/train'))
    valid_df['td_jpg_path'] = valid_df['image_path'].apply(lambda x : os.path.dirname(x).replace('/all','/val'))
    test_df['td_jpg_path'] = test_df['image_path'].apply(lambda x : os.path.dirname(x).replace('/all','/test'))

    train_df['td_json_paths'] = train_df['json_paths'].apply(lambda x : os.path.dirname(x).replace('/all','/train'))
    valid_df['td_json_paths'] = valid_df['json_paths'].apply(lambda x : os.path.dirname(x).replace('/all','/val'))
    test_df['td_json_paths'] = test_df['json_paths'].apply(lambda x : os.path.dirname(x).replace('/all','/test'))


    train_df.to_csv(os.path.join(OUTPUT_DIR, f'{date_prefix}_train_df.csv'),encoding='euc-kr')
    valid_df.to_csv(os.path.join(OUTPUT_DIR, f'{date_prefix}_valid_df.csv'),encoding='euc-kr')
    test_df.to_csv(os.path.join(OUTPUT_DIR, f'{date_prefix}_test_df.csv'),encoding='euc-kr')

    # # # #$!@$$$$$$$$$$$$$$$$$$$$$$$
    # test_df = pd.read_csv(os.path.join(OUTPUT_DIR, f'{date_prefix}_test_df.csv'), encoding='euc-kr', index_col=0)
    # valid_df = pd.read_csv(os.path.join(OUTPUT_DIR, f'{date_prefix}_valid_df.csv'), encoding='euc-kr', index_col=0)
    # train_df = pd.read_csv(os.path.join(OUTPUT_DIR, f'{date_prefix}_train_df.csv'), encoding='euc-kr', index_col=0)

    # # # test 복사
    # # copy_files_in_parallel(test_df['image_path'].to_list(), test_df['td_jpg_path'].to_list())
    # copy_files_in_parallel(test_df['json_paths'].to_list(), test_df['td_json_paths'].to_list())
    # print('finished test')

    # ## valid 복사
    # # copy_files_in_parallel(valid_df['image_path'].to_list(), valid_df['td_jpg_path'].to_list())
    # copy_files_in_parallel(valid_df['json_paths'].to_list(), valid_df['td_json_paths'].to_list())
    # print('finished valid')

    # ## train 복사
    # # copy_files_in_parallel(train_df['image_path'].to_list(), train_df['td_jpg_path'].to_list())
    # copy_files_in_parallel(train_df['json_paths'].to_list(), train_df['td_json_paths'].to_list())
    # print('finished train')
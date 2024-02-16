# 원예식물 화분류 물주기 수분공급 주기 생육데이터
- 인공지능 학습을 통해 최적의 주거 환경(다세대주택 등)에서의 원예식물의 상태(잎, 줄기, 뿌리 등)를 포함한 물주기(수분공급)의 데이터셋을 구축
    - 원예식물 물주기 최적화와 생육 상태 탐지 및 생육 예측이 가능하도록 알고리즘 설계 및 데이터베이스 구축
    
## 목차
1. [구축데이터정보](#구축데이터정보)
1. [모델사용설명서](#모델사용설명서)
1. [환경설치가이드](#환경설치가이드)
1. [도커(.tar)사용방법](#도커사용방법)
1. [라이센스 정보](#License)

-----------------------------------------------------------------------------

## 구축데이터정보
- 파일명 구성 정보
    - 예시 파일명 : 
        - N50-B-3-15-L-3-V-230507-001041.json

        |순서|분류|정보|비고|
        |--|--|--|--|
        |1|세부과제ID|50.원예식물 물주기 생육데이터|1개|
        |2|대분류|화초(A), 관상수(B)|2종류|
        |3|중분류|건생식물(1), 중생식물(2), ..., 수생식물(4)|4종류|
        |4|소분류|스투키(01), 선인장(02), ... 디펜바키아(15)|15종류|
        |5|장소|거실(L), 베란다(B)|2종류|
        |6|세분류|건조(1), 일반(2), 과습(3)|3종류|
        |7|촬영각도|Vertical(V), Horizontal(H)|2종류|
        |8|수집날짜|2023년05월07일|--|
        |9|일련번호|000001 ~ |--|
        |10|파일형식|jpg, json|2종류|

### 분할 데이터 구성 비율 및 모델별 상세 파일수
- Train/Validation/Test 8:1:1

#### 이미지분류 모델
| LrgCat | MidCat | SmlCat    | classID | train  | val   | test  | total  |
|--------|--------|-----------|---------|--------|-------|-------|--------|
| A.화초   | 1.건생식물 | 01.스투키    | A-1-01  | 25961  | 3246  | 3245  | 32452  |
| A.화초   | 1.건생식물 | 02.선인장    | A-1-02  | 27960  | 3495  | 3495  | 34950  |
| A.화초   | 1.건생식물 | 03.금전수    | A-1-03  | 28088  | 3511  | 3511  | 35110  |
| A.화초   | 2.중생식물 | 04.테이블야자  | A-2-04  | 27493  | 3437  | 3437  | 34367  |
| A.화초   | 2.중생식물 | 05.홍콩야자   | A-2-05  | 28192  | 3524  | 3524  | 35240  |
| A.화초   | 2.중생식물 | 06.호접란    | A-2-06  | 24864  | 3108  | 3108  | 31080  |
| A.화초   | 2.중생식물 | 07.스파티필럼  | A-2-07  | 25356  | 3170  | 3170  | 31696  |
| A.화초   | 3.습생식물 | 08.보스턴고사리 | A-3-08  | 25462  | 3183  | 3183  | 31828  |
| A.화초   | 3.습생식물 | 09.몬스테라   | A-3-09  | 24296  | 3037  | 3037  | 30370  |
| A.화초   | 4.수생식물 | 10.부레옥잠   | A-4-10  | 24467  | 3059  | 3058  | 30584  |
| B.관상수  | 1.건생식물 | 11.올리브나무  | B-1-11  | 27296  | 3412  | 3412  | 34120  |
| B.관상수  | 2.중생식물 | 12.오렌지쟈스민 | B-2-12  | 28002  | 3501  | 3500  | 35003  |
| B.관상수  | 2.중생식물 | 13.관음죽    | B-2-13  | 27560  | 3446  | 3445  | 34451  |
| B.관상수  | 3.습생식물 | 14.벵갈고무나무 | B-3-14  | 27032  | 3380  | 3379  | 33791  |
| B.관상수  | 3.습생식물 | 15.디펜바키아  | B-3-15  | 24253  | 3032  | 3032  | 30317  |
| ---      | ---       | ---           | ---      | 396282 | 49541 | 49536 | 495359 |

#### 생장예측 모델
| LrgCat | MidCat | SmlCat    | classID | train  | val   | test  | total  |
|--------|--------|-----------|---------|--------|-------|-------|--------|
| A.화초   | 1.건생식물 | 01.스투키    | A-1-01  | 21143  | 5013  | 6296  | 32452  |
| A.화초   | 1.건생식물 | 02.선인장    | A-1-02  | 27916  | 745   | 6289  | 34950  |
| A.화초   | 1.건생식물 | 03.금전수    | A-1-03  | 35110  | 0     | 0     | 35110  |
| A.화초   | 2.중생식물 | 04.테이블야자  | A-2-04  | 28574  | 0     | 5793  | 34367  |
| A.화초   | 2.중생식물 | 05.홍콩야자   | A-2-05  | 23535  | 5970  | 5735  | 35240  |
| A.화초   | 2.중생식물 | 06.호접란    | A-2-06  | 25654  | 0     | 5426  | 31080  |
| A.화초   | 2.중생식물 | 07.스파티필럼  | A-2-07  | 21343  | 4752  | 5601  | 31696  |
| A.화초   | 3.습생식물 | 08.보스턴고사리 | A-3-08  | 27814  | 4014  | 0     | 31828  |
| A.화초   | 3.습생식물 | 09.몬스테라   | A-3-09  | 21479  | 4709  | 4182  | 30370  |
| A.화초   | 4.수생식물 | 10.부레옥잠   | A-4-10  | 23762  | 5600  | 1222  | 30584  |
| B.관상수  | 1.건생식물 | 11.올리브나무  | B-1-11  | 27690  | 689   | 5741  | 34120  |
| B.관상수  | 2.중생식물 | 12.오렌지쟈스민 | B-2-12  | 23767  | 5635  | 5601  | 35003  |
| B.관상수  | 2.중생식물 | 13.관음죽    | B-2-13  | 34451  | 0     | 0     | 34451  |
| B.관상수  | 3.습생식물 | 14.벵갈고무나무 | B-3-14  | 28160  | 5631  | 0     | 33791  |
| B.관상수  | 3.습생식물 | 15.디펜바키아  | B-3-15  | 30317  | 0     | 0     | 30317  |
| ---      | ---       | ---           | ---      | 400715 | 42758 | 51886 | 495359 |


### 폴더 구성
- 모델별, 분할된 데이터 별 폴더구조는 대동소이 하며 예시는 다음과 같다.
    ```bash
    # 전체 구조 예시(식물 이미지 분류 모델의 테스트셋 기준)
    이미지분류/
    ├── train
    │   └──...
    ├── val
    │   └──...
    └── test
        ├── 라벨링데이터
        │   └──...
        └── 이미지데이터
            ├── A.화초
            │   ├── 1.건생식물
            │   │   ├── 01.스투키
            │   │   ├── 02.선인장
            │   │   └── 03.금전수
            │   ├── 2.중생식물
            │   │   ├── 04.테이블야자
            │   │   ├── 05.홍콩야자
            │   │   ├── 06.호접란
            │   │   └── 07.스파티필럼
            │   ├── 3.습생식물
            │   │   ├── 08.보스턴고사리
            │   │   └── 09.몬스테라
            │   └── 4.수생식물
            │       └── 10.부레옥잠
            └── B.관상수
                ├── 1.건생식물
                │   └── 11.올리브나무
                ├── 2.중생식물
                │   ├── 12.오렌지쟈스민
                │   └── 13.관음죽
                └── 3.습생식물
                    ├── 14.벵갈고무나무
                    └── 15.디펜바키아
                        ├── N50-B-3-15-B-1-H-230825-000132.jpg
                        ...
                        └── N50-B-3-15-L-3-V-231018-000890.jpg

    # 폴더 깊이별 구성 폴더 규칙 다음과 같다
    <모델명>/
    └── <데이터분할명>
        └── <데이터종류>
            └── <대분류>
                └── <중분류>
                    └── <소분류>
                        └── <파일명>
    ```

-----------------------------------------------------------------------------

## 모델사용설명서
- 모델별 상세 설명


### 식물 이미지 분류

#### 모델 목적
- 식물 이미지를 통해 식물 종 분류
<img src="https://github.com/y0ngma/VirtualMachine/blob/main/horticulture/readme_insert_files/test_result/4_cf_matrix.png" width="500px" height="500px" title="cf"/>

#### 평가 지표
- F1-점수
    <img src="https://github.com/y0ngma/horticulture/blob/main/readme_insert_files/criteria_classify.png" title="이미지분류 평가지표"/>

#### 모델 버전
- 2.0

#### 학습 알고리즘 및 프레임워크
- PolyNet, PyTorch, Scikit-Learn

#### 학습 조건
* Image Data Generator
    - rescale: 1./255  #픽셀 값을 0과 1 사이로 정규화
    - rotation_range: 90  #무작위 회전 (0-20도)
    - width_shift_range: 0.2, #수평 이동
    - height_shift_range: 0.2  #수직 이동
    - shear_range: 0.2  #전단 변환
    - zoom_range: 0.2  #무작위 줌
    - horizontal_flip: True  #수평 뒤집기
    - fill_mode: 'nearest'  #새로 생성된 픽셀 채우기 방식
* PolyNet Algorithm
    - activation: 'softmax’
    - batch_size: 32
    - Dropout: 0.4
    - loss: ‘categorical_crossentropy’
    - epoch: 1
    - learning_rate: 0.0004


### 식물 생장 예측

#### 모델 목적
- 식물 센서 데이터를 통해 생장 예측

#### 평가 지표
- Mean Absolute Percentage Error(평균절대비율오차)
    <img src="https://github.com/y0ngma/horticulture/blob/main/readme_insert_files/criteria_predict.png" title="생장예측 평가지표"/>

#### 모델 버전
- 1.0

#### 학습 알고리즘 및 프레임워크
- RandomForest, PyTorch, Scikit-Learn

#### 학습 조건
* RandomForest 파라미터 범위
    - 'n_estimators': [100, 200],
    - 'max_depth': [None, 10],
    - 'min_samples_split': [2, 5],
    - 'min_samples_leaf': [1, 2]
* GridsearchCV를 활용한 최적의 파라미터 선정
    - n-jobs=-1 (모든 코어 사용)
    - cv=3 (교차검증을 위한 fold 수)


### 모델 소스 코드 설명

- 도커 구동시 루트경로에 다음과 같은 소스코드 및 각종 파일 확인가능
    ```
    (docker container 내부)
    ├── /app
    │   ├── classify_test.py
    │   ├── classify_train.py
    │   ├── plant_prediction_test.py
    │   ├── plant_prediction_train.py
    │   ├── collect_info.sh
    │   └── data_split811.py
    └── /mnt
        ├── dataset
        │   ├── 생장예측
        │   │   └── ...
        │   └── 이미지분류
        │       ├── train
        │       ├── val
        │       └── test
        └── output
            ├── img_model_2024-01-09.h5
            └── 2023-12-29_prediction_model.abc
    ```

- 파일별 세부 용도는 다음과 같다

    |파일명|쓰임새|비고|순서|
    |--|--|--|--|
    |classify_train.py|식물 이미지분류모델 학습|/mnt/output경로에 가중치 저장</br>/mnt/dataset내 train/ 및 val/ 필요|1-1|
    |img_model_2024-01-09.h5|식물 이미지분류모델의 학습완료된 가중치|classify_train.py 실행 종료시</br>/mnt/output/내에 생성|1-2|
    |classify_test.py|식물 이미지분류모델 성능 측정|/mnt/output경로의 가중치 및</br>/mnt/dataset/test/ 필요|1-3|
    |plant_prediction_train.py|식물 생장예측모델 학습|/mnt/output경로에 가중치 저장</br>/mnt/dataset내 train/ 및 val/ 필요|2-1|
    |2023-12-29_prediction_model.abc|식물 생장예측모델의 학습완료된 가중치|plant_prediction_train.py 실행 종료시</br>/mnt/output/내에 생성|2-2|
    |plant_prediction_test.py|식물 생장예측모델 성능 측정|/mnt/output경로의 가중치 및</br>/mnt/dataset/test/ 필요|2-3|
    |collect_info.sh|하드웨어정보 출력파일|실행 필요 없음|---|
    |data_split811.py|식물 이미지분류모델 데이터분할한 코드로써 분류기준 참고가능|실행 필요 없음|---|


-----------------------------------------------------------------------------

## 환경설치가이드

### 개발 시스템 환경
1. 하드웨어 정보

    |분류|정보|
    |--|--|
    |CPU 정보|12th Gen Intel(R) Core(TM) i7-12700K|
    |운영체제|Ubuntu 20.04.4 LTS|
    |HDD 정보 (GB)|14|
    |RAM 정보 (GB)|31|
    |GPU 정보|NVIDIA Corporation GA104 [GeForce RTX 3060 Ti Lite Hash Rate] (rev a1)|
    |그래픽드라이버 버전|515.105.01|
    |CUDA 버전|11.2.1|

2. 개발 프레임워크 버전 정보
    - 개발 언어
        - Python 3.8.10
    - 프레임워크
        - tensorflow==2.10.0
        - keras==2.10.0
        - scikit-learn==0.24.2
        - numpy==1.23.2


### 환경 구축 방법
- Dockerfile 과 docker-compose.yaml 파일에 기반하여 직접 이미지를 생성

#### 1. 데이터셋 등 마운트 경로 설정
- docker-compose.yml 파일내에 마운트하고자 하는 로컬경로 수정
- `< ~ 로컬경로>` 부분을 자유롭게 수정하여 사용합니다.
    ```bash
        ...
        volumes:
            - <컨테이너 내에서 실행할 소스코드가 있는 로컬경로>:/app
            - <컨테이너 내에서 생성된 산출물을 확인할 로컬경로>:/mnt/output
            - <컨테이너 내에서 사용할 실데이터가 있는 로컬경로>:/mnt/dataset
        ...
    ```
- 예시 : 아래의 경우 <컨테이너 내에서 사용할 실데이터가 있는 로컬경로>는 /path/to/dataset 이다.
    ```
    /path/to/dataset
                ├── 생장예측
                └── 이미지분류
                    ├── train
                    ├── val
                    └── test
                        ├── 라벨링데이터
                        └── 이미지데이터
    ```

#### 2. 이미지 생성 및 컨테이너 구동
- docker-compose.yml 이 있는 경로로 이동 후 다음 명령어 실행
    ```bash
    docker compose up -d --build
    ```

#### 3. 컨테이너 접속
- 구동중인 컨테이너명을 확인하여 작성
    ```bash
    # 출력물 중 NAMES 확인
    docker ps -f name=horticulture_container

    # 해당 내용을 아래에 기입(예:horticulture_container)
    docker exec -it horticulture_container bash
    ```

#### 4. 소스코드 실행
- 접속한 컨테이너의 CLI(터미널)로 실행하고자 하는 소스코드 실행
    ```bash
    # 식물이미지분류모델 학습 및 검증 코드 실행예시
    python classify_train.py
    python classify_test.py

    # 식물생장예측모델 학습 및 검증 코드 실행예시
    python plant_prediction_train.py
    python plant_prediction_test.py
    ```
- 학습코드 실행시 `<컨테이너 내에서 생성된 산출물을 확인할 로컬경로>` 경로에 가중치가 저장됨
- 제공된 모델 파일에 대해 검증 코드를 실행하고자 할 때에는 상기 경로에 해당 파일을 먼저 옮김
    - img_model_2024-01-09.h5
    - 2023-12-29_prediction_model.abc

#### (참고)tensorflow gpu 인식 확인방법
- 정보출력방법 : (컨테이너내) 커맨드라인에서 `python`입력 후 다음을 입력. (예시 출력값 참고) 
    ```py
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    # 예시 출력값
    [name: "/device:CPU:0"...
    ,name: "/device:GPU:0"..., name: Tesla T4, ...]

    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.test.is_built_with_cuda())
    # 예시 출력값
    Num GPUs Available: 4
    True
    ```

-----------------------------------------------------------------------------

## 도커(.tar) 사용한 모델 검증 방법
- Dockerfile 을 기반하여 생성된 .tar파일을 `docker load`를 통해 docker image를 로드하여 미리 구축된 환경내에서 작업
    - 본 절차는 모델학습이 아니라 제공된 학습완료 가중치의 성능검증 재현을 위한것임
- 모델별 명령어는 다소 차이가 있으나, 절차는 다음과 동일하다.
    1. 압축파일(.tar)경로로 이동 후 도커 이미지 로드
    1. 도커 컨테이너 실행 및 접속
    1. 검증코드 실행

### 생장예측 모델 검증
```bash
#### 1. 우선 .tar가 있는 경로로 이동후 다음 명령어 실행
# tarfile로 이미지 로드(좀걸림)
docker load -i ./nia50-json.tar


#### 2. 컨테이너 실행 후 진입
# 명령어 작성방법
docker run -it \
    -v <산출물확인 경로>:/mnt/output \
    -v <시험용데이터셋 경로>:/mnt/dataset \
    --gpus all \
    --name nia50-json-test \
    nia50-json:1.0 bash

# 명령어 작성예시
docker run -it \
    -v /mnt/nia50/final_output:/mnt/output \
    -v /mnt/nia50/final/생장예측/test/라벨링데이터:/mnt/dataset \
    --gpus all \
    --name nia50-json-test \
    nia50-json:1.0 bash


#### 3. 컨테이너 터미널 안에서 소스코드 파일 실행
# 식물 생장예측모델 검증 코드 실행예시
python plant_prediction_test.py
```

### 이미지분류 모델 검증
```bash
#### 1. 우선 .tar가 있는 경로로 이동후 다음 명령어 실행
# tarfile로 이미지 로드(좀걸림)
docker load -i ./nia50-jpg2.tar


#### 2. 컨테이너 실행 후 진입
# 명령어 작성방법
docker run -it \
    -v <산출물확인 경로>:/mnt/output \
    -v <이미지데이터 경로>:/mnt/dataset/이미지분류/test/이미지데이터 \
    -v <라벨링데이터 경로>:/mnt/dataset/이미지분류/test/라벨링데이터 \
    --gpus all \
    --name nia50-test \
    nia50-jpg:2.0 bash

# 명령어 작성예시
docker run -it \
    -v /mnt/nia50/final_output:/mnt/output \
    -v /mnt/nia50/final/이미지분류/test/이미지데이터:/mnt/dataset/이미지분류/test/이미지데이터 \
    -v /mnt/nia50/final/이미지분류/test/라벨링데이터:/mnt/dataset/이미지분류/test/라벨링데이터 \
    --gpus all \
    --name nia50-test \
    nia50-jpg:2.0 bash


#### 3. 컨테이너 터미널 안에서 소스코드 파일 실행
# 식물이미지분류모델 검증 코드 실행예시
python classify_test.py
```


---
- 깃허브주소 https://github.com/y0ngma/horticulture

## License
Copyright (c) 2023 Gnewsoft SPDX-License-Identifier: MIT

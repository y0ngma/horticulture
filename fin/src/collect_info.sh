#!/bin/bash
# 호스트머신에서 접근 할 수 있도록 마운트 된 디렉토리로 파일경로설정
FILEPATH=/mnt/output/version_report.txt

# '>'는 overwrite로 기존것 대체, '>>'는 append로 단순 한줄추가
echo "주의 - 호스트머신과 컨테이너 내 그래픽카드 버전일치 확인" > $FILEPATH

# 특정명령어를 필터링한 후 'awk'으로 '콜론과 공백한번이상반복'으로 분리된 두 번째 필드출력
CPU_NAME=$(lscpu | grep 'Model name' | awk -F ':[[:space:]]+' '{print $2}')
echo "CPU 정보=$CPU_NAME" >> $FILEPATH

OS_INFO=$(lsb_release -a | grep 'Description' | awk -F ':[[:space:]]+' '{print$2}')
echo "운영체제=$OS_INFO" >> $FILEPATH

# 루트경로의 출력행 중 두 번째 행, 네 번째 열선택
HDD_INFO=$(df -h / | awk 'NR==2{print $4}')
echo "HDD 정보 (GB)=$HDD_INFO" >> $FILEPATH

RAM_INFO=$(free -h | grep Mem: | awk '{print $2}')
echo "RAM 정보 (GB)=$RAM_INFO" >> $FILEPATH

GPU_NAME=$(lspci | grep VGA | awk -F ':[[:space:]]+' '{print $2}')
echo "GPU 정보=$GPU_NAME" >> $FILEPATH
# nvcc --version과 nvidia-smi의 출력된 버전이 다를 수 있음.
GRAPHICS_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)
echo "그래픽드라이버 버전=$GRAPHICS_DRIVER_VERSION" >> $FILEPATH

CUDA_VERSION=$(env | grep CUDA_VERSION | awk -F '=' '{print $2}')
echo "CUDA 버전=$CUDA_VERSION" >> $FILEPATH

# echo "cuDNN Version:" >> $FILEPATH
# cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2 >> $FILEPATH
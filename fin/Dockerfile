FROM tensorflow/tensorflow:2.10.0-gpu-jupyter

ENV TZ=Asia/Seoul

RUN apt-get update && \
    # 각종 유틸 설치
    apt-get -y install sudo vim tree lsb pciutils tzdata && \
    update-pciids && \
    # 시간설정 심볼릭링크 연결 및 쓰기
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    # 나눔글꼴(fonts-nanum)을 설치하고, fc-cache 명령으로 폰트 캐시 삭제
    apt-get -y install fonts-nanum* && \
    apt install -y fontconfig && \
    fc-cache -fv && \
    # 나눔 글꼴을 matplotlib 에 복사하고, matplotlib의 폰트 캐시를 삭제 
    cp /usr/share/fonts/truetype/nanum/Nanum* /usr/local/lib/python3.8/dist-packages/matplotlib/mpl-data/fonts/ttf/ && \
    rm -rf /home/ubuntu/.cache/matplotlib/* && \
    # 각종 라이브러리 설치
    pip3 install --upgrade pip && \
    pip install pandas seaborn missingno openpyxl tqdm scikit-learn==0.24.2 && \
    pip install tensorflow-hub==0.15.0 protobuf==3.19.6 && \
    # opencv
    apt-get -y install libgl1-mesa-glx && \
    pip install opencv-python

# 컨테이너 내 작업 디렉토리를 설정
WORKDIR /app

# Docker 컨테이너를 실행할 때 실행될 명령을 설정합니다.
CMD ["tail","-f","/dev/null"]
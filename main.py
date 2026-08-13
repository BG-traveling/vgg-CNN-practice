#라벨만들기 위한 os, json
import os, json, re
import matplotlib.pyplot as plt

#비전 관련 작업 cv로 진행할 때, numpy를 함께 임포트
import cv2
import numpy as np

import shutil #.sh / .bash 

from Utils.visualize import show_yolo_label
from Preprocessing.yolo_preprocessing import label_from_json, label_from_txt, create_yolo_label, move_datas, move_label_datas
from Utils import augmentation as aug


# 생성 구조 (Ultralytics 표준):
#   Data/PeachDataset/yolo_dataset/
#     images/train/*.jpg
#     images/valid/*.jpg
#     labels/train/*.txt
#     labels/valid/*.txt

#YOLO 기술문서
#https://docs.ultralytics.com/modes/train,,
if __name__ == '__main__':
    #실행 연습
    # fig, ax = plt.subplots(1, 2)
    # image = r'./Data/PeachDataset/peach_image/train/A220120XX_10306.jpg'
    # image = cv2.imread(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # ax[0].imshow(image)
    # image, label = aug.flip_horizontal(image, None)
    # ax[1].imshow(image)
    # plt.show()
    # print(label)

    aug.pipe_augmentation()






#딥러닝 시퀀스
#1.데이터 가져오기
#2.데이터 정제(preprocessing)
#3.알고리즘선택
#4.훈련
#5.검증
#6.평가
#7.배포
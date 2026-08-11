#라벨만들기 위한 os, json
import os, json
import matplotlib.pyplot as plt
import cv2
import re

if __name__ == '__main__':
    sample_label_path = r'C:\kdh\머신러닝, 딥러닝\로컬1일차_샘플\Data\PeachDataset\peach_label\A220120XX_10306.json'

    with open(sample_label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        words = []

        for line in lines:
            #공백제거
            parts = line.strip().split()
            words.append([re.sub(r'[^a-zA-Z0-9.]', '', x) for x in parts])
        print(words)
            


#딥러닝 시퀀스
#1.데이터 가져오기
#2.데이터 정제(preprocessing)
#3.알고리즘선택
#4.훈련
#5.검증
#6.평가
#7.배포
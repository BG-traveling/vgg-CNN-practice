#모델 Zoo : https://docs.pytorch.org/serve/model_zoo.html

#컴퓨터(os)와 이미지 데이터 로딩
import os 
from PIL import Image
import matplotlib.pyplot as plt

#신경망 구성 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim

#데이터셋, 데이터 로더, 데이터로딩 시 스플릿 관련 라이브러리
from torchvision import datasets, models, transforms 
from torch.utils.data import DataLoader, Dataset, random_split

#전이학습 모델 가중치
from torchvision.models import vgg11, VGG11_Weights

#내가 만든 파일, 폴더에서 함수 불러오기
from Utils.DataLoader import get_dataloader
from models.vgg import get_vgg_model
import train #파일 전체를 import해서 함수를 쓸 때 소속을 train. 으로 시작
from eval import evaluate

def run_epoch(model, history, EPOCHS=10):

    #꼭 해야 할 것 -> GPU를 쓰기 위해!
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #에포크마다 train_one_epoch랑 eval 함수를 돌리면 됨!
    #오차함수, 최적화함수, 히스토리 딕셔너리 추가, 에포크 정해주기
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    best_acc = 0.0

    for e in range(EPOCHS):
            #평균 loss, 맞춘 비율 %
            #model, loader, criterion, optimizer, device
            train_loss, train_acc = train.train_one_epoch(model, 
                                                    train_loader,
                                                    criterion=criterion,
                                                    optimizer=optimizer,
                                                    device=device)
            valid_loss, valid_acc = evaluate(model, 
                                                    valid_loader,
                                                    criterion=criterion,
                                                    optimizer=optimizer,
                                                    device=device)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['valid_loss'].append(valid_loss)
            history['valid_acc'].append(valid_acc)
    
            if valid_acc > best_acc :
                best_acc = valid_acc
                save_path = f'./CIFAR10_VGG11_{e+1}epoch_{valid_acc:.2f}.pth' 
                torch.save(model.state_dict(), save_path)
    
                print(f'최고 기록 ! {e+1}회 에포크 --> {valid_acc:.2f}')
    return history
    

if __name__ == '__main__':
    #get_dataloader = 데이터를 로딩하는 함수 
    train_loader, valid_loader = get_dataloader()

    #vgg11모델과 가중치를 로딩하는 함수
    model = get_vgg_model()
    history = {'train_loss':[], 'train_acc':[], 'valid_loss':[], 'valid_acc':[]}

    #model_history 훈련이 끝났을 때의 총 history를 반환
    model_history = run_epoch(model, history, EPOCHS=50)
from preprocess import Data_Processor
from model import *
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Set arguments.")
    parser.add_argument("--data_path", default="./data/sns_dataset.csv", type=str, help="Traing data path")
    parser.add_argument("--model_path", default="./model/model.pkl", type=str, help="Save model path")
    #parser.add_argument("--log_path", default="./model/model.pkl", type=str, help="Save model path")
    
    args = parser.parse_args()
    
    # data_path = "./data/sns_dataset.csv"
    # model_path = "./model/model.pkl"
    
    # 데이터 불러오기 & 전처리
    dp = Data_Processor(data_path=args.data_path)
    train_texts, val_texts, train_labels, val_labels = dp.data_preproces()
    
    # 모델 학습
    vect, nb = train_model(train_texts, train_labels)
    
    # 모델 저장
    save_model((vect, nb), args.model_path)
    
    # 모델 평가
    eval_model(vect, nb, val_texts, val_labels)
    
    print("End Training Model")
    #python train.py --data_path ./data/sns_dataset.csv --model_path ./model/model.pkl

    
    """
    모델 학습
    - 데이터 불러오기 & 전처리
    - 모델 학습
    - 모델 저장
    - 모델 평가
    
    모델 예측
    - 예측 데이터 불러오기 & 전처리
    - 모델 불러오기
    - 예측 수행
    - 결과 저장
    """
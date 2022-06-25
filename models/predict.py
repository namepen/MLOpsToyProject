from preprocess import Data_Processor
from model import *
import argparse



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Set arguments.")
    parser.add_argument("--file_path", default="./data/sns_dataset_test.csv", type=str, help="predict dataframe path")
    parser.add_argument("--model_path", default="./model/model.pkl", type=str, help="model path")
    parser.add_argument("--save_path", default="./data/result.csv", type=str, help="result save path")
    
    args = parser.parse_args()
    
    test_predict(args.file_path, args.model_path, args.save_path)
    # python predict.py --file_path ./data/sns_dataset_test.csv --model_path ./model/model.pkl --save_path ./data/result.csv
    
    print("End predict")


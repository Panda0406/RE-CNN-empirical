# -*- coding: utf-8 -*-
import argparse

def load_hyperparameters():
    parser = argparse.ArgumentParser(description='Multi-window CNN for SemEval2010-task-8 relation classification')

    parser.add_argument("--window_size", default=[3,4,5], type=list)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--kernel_num", default=100, type=float)
    parser.add_argument("--max_pos", default=30, type=int, help="The maximum value of position feature.")
    parser.add_argument("--max_sent", default=110, type=int, help="The maximum length of input sentence.")
    parser.add_argument("--pos_dim", default=10, type=int, help="The dimension of position feature")
    parser.add_argument("--label_num", default=19, type=int)
    parser.add_argument("--min_freq", default=5000, type=int)

    parser.add_argument("--batch_size", default=30, type=int)
    parser.add_argument("--epoch_max", default=15, type=int)
    parser.add_argument("--lr", default=0.5, type=float)

    parser.add_argument("--model_save_path", default='./models/', type=str)
    parser.add_argument("--result_save_path", default='./Answers/', type=str)

    parser.add_argument("--seed", default='0', type=int)
    parser.add_argument("--device", default='0', type=int)

    args = parser.parse_args()

    print("------HYPERPARAMETERS-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))
    print("----------------------------")

    return args

if __name__ == "__main__":
    load_hyperparameters()
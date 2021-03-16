import math
from functools import reduce
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model.tests.test_perceptron import random_state
from sklearn.preprocessing import OrdinalEncoder
from sksurv.datasets import load_gbsg2, load_veterans_lung_cancer, load_whas500
from sksurv.metrics import concordance_index_censored
from sksurv.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
from sksurv.tree.tree import _array_to_step_function
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
import os


def start_computation(path_root, dataset, duration_col, event_col):
    os.mkdir(path_root)
    train_set, test_set = split_test_set(dataset)
    compute_2_even(train_set, test_set, path_root, duration_col, event_col)
    compute_2_uneven(train_set, test_set, path_root, duration_col, event_col)
    compute_5_even(train_set, test_set, path_root, duration_col, event_col)
    compute_5_uneven(train_set, test_set, path_root, duration_col, event_col)


def compute_2_even(train_set, test_set, path, duration_col, event_col):
    all_2_even_data = split_2_even(train_set, test_set)
    os.mkdir(path + "/2_even")
    current_client = 1
    for client_data in all_2_even_data:
        current_dir_path = path + "/2_even/client" + str(current_client)
        os.mkdir(current_dir_path)

        train_set = client_data[0]
        path_train = "train_" + str(current_client) + "_2_even.csv"
        train_set.to_csv(current_dir_path + "/" + path_train, index=False)

        test_set = client_data[1]
        path_test = "test_" + str(current_client) + "_2_even.csv"
        test_set.to_csv(current_dir_path +  "/" +path_test, index=False)
        create_config(path_train, path_test, duration_col, event_col, current_dir_path)

        current_client += 1

def compute_2_uneven(train_set, test_set, path, duration_col, event_col):
    all_2_uneven_data = split_2_uneven(train_set, test_set)
    os.mkdir(path + "/2_uneven")
    current_client = 1
    for client_data in all_2_uneven_data:
        current_dir_path = path + "/2_uneven/client" + str(current_client)
        os.mkdir(current_dir_path)

        train_set = client_data[0]
        path_train = "train_" + str(current_client) + "_2_uneven.csv"
        train_set.to_csv(current_dir_path + "/" + path_train, index=False)

        test_set = client_data[1]
        path_test = "test_" + str(current_client) + "_2_uneven.csv"
        test_set.to_csv(current_dir_path +  "/" +path_test, index=False)
        create_config(path_train, path_test, duration_col, event_col, current_dir_path)

        current_client += 1

def compute_5_even(train_set, test_set, path, duration_col, event_col):
    all_5_even_data = split_5_even(train_set, test_set)
    os.mkdir(path + "/5_even")
    current_client = 1
    for client_data in all_5_even_data:
        current_dir_path = path + "/5_even/client" + str(current_client)
        os.mkdir(current_dir_path)

        train_set = client_data[0]
        path_train = "train_" + str(current_client) + "_5_even.csv"
        train_set.to_csv(current_dir_path + "/" + path_train, index=False)

        test_set = client_data[1]
        path_test = "test_" + str(current_client) + "_5_even.csv"
        test_set.to_csv(current_dir_path +  "/" +path_test, index=False)
        create_config(path_train, path_test, duration_col, event_col, current_dir_path)

        current_client += 1


def compute_5_uneven(train_set, test_set, path, duration_col, event_col):
    all_5_uneven_data = split_5_uneven(train_set, test_set)
    os.mkdir(path + "/5_uneven")
    current_client = 1
    for client_data in all_5_uneven_data:
        current_dir_path = path + "/5_uneven/client" + str(current_client)
        os.mkdir(current_dir_path)

        train_set = client_data[0]
        path_train = "train_" + str(current_client) + "_5_uneven.csv"
        train_set.to_csv(current_dir_path + "/" + path_train, index=False)

        test_set = client_data[1]
        path_test = "test_" + str(current_client) + "_5_uneven.csv"
        test_set.to_csv(current_dir_path +  "/" +path_test, index=False)
        create_config(path_train, path_test, duration_col, event_col, current_dir_path)

        current_client += 1

def split_test_set(dataset):
    train_set = dataset.sample(frac=0.75, random_state=random_state)
    test_set = dataset.drop(train_set.index)

    print('train')
    print(len(train_set))
    # print(train_set.head())
    # train_set.to_csv("rossi_train.csv", index=False)

    print('\ntest')
    print(len(test_set))
    # print(test_set.head())
    # test_set.to_csv("rossi_test.csv", index=False)
    return train_set, test_set


def split_2_even(train_set, test_set):
    split_train = round(len(train_set) / 2)
    split_test = round(len(test_set) / 2)

    train_1 = train_set.iloc[:split_train, :]
    train_2 = train_set.iloc[split_train:, :]

    # train_1.to_csv("train1.csv", index=False)
    # train_2.to_csv("train2.csv", index=False)

    test_1 = test_set.iloc[:split_test, :]
    test_2 = test_set.iloc[split_test:, :]

    # test_1.to_csv("test1.csv", index=False)
    # test_2.to_csv("test2.csv", index=False)

    # print('\n\ntrain')
    # print(len(train_set))
    # print(len(train_1))
    # print(len(train_2))

    # print('\ntest')
    # print(len(test_set))
    # print(len(test_1))
    # print(len(test_2))

    return [[train_1, test_1], [train_2, test_2]]


def split_2_uneven(train_set, test_set):
    split_train = round(len(train_set) / 4)
    split_test = round(len(test_set) / 4)

    train_1 = train_set.iloc[:split_train, :]
    train_2 = train_set.iloc[split_train:, :]

    test_1 = test_set.iloc[:split_test, :]
    test_2 = test_set.iloc[split_test:, :]

    print('\n\ntrain')
    print(len(train_set))
    print(len(train_1))
    print(len(train_2))

    print('\ntest')
    print(len(test_set))
    print(len(test_1))
    print(len(test_2))

    return [[train_1, test_1], [train_2, test_2]]


def split_5_even(train_set, test_set):
    # TODO: nicht ganz even wenn zahl nicht teilbar
    split_train_1 = math.floor(len(train_set) / 5)
    split_train_2 = 2 * split_train_1
    split_train_3 = 3 * split_train_1
    split_train_4 = 4 * split_train_1

    split_test_1 = math.floor(len(test_set) / 5)
    split_test_2 = 2 * split_test_1
    split_test_3 = 3 * split_test_1
    split_test_4 = 4 * split_test_1

    train_1 = train_set.iloc[0:split_train_1]
    train_2 = train_set.iloc[split_train_1:split_train_2]
    train_3 = train_set.iloc[split_train_2:split_train_3]
    train_4 = train_set.iloc[split_train_3:split_train_4]
    train_5 = train_set.iloc[split_train_4:, :]

    test_1 = test_set.iloc[0:split_test_1]
    test_2 = test_set.iloc[split_test_1:split_test_2]
    test_3 = test_set.iloc[split_test_2:split_test_3]
    test_4 = test_set.iloc[split_test_3:split_test_4]
    test_5 = test_set.iloc[split_test_4:, :]

    # test_1 = test_set.iloc[:split_test, :]
    # test_2 = test_set.iloc[split_test:, :]

    print('\n\ntrain')
    print(len(train_set))
    # print(len(train_1))
    # print(len(train_2))

    print(split_train_1)
    print(len(train_1))
    print("\n")
    print(split_train_2)
    print(len(train_2))
    print("\n")
    print(split_train_3)
    print(len(train_3))
    print("\n")
    print(split_train_4)
    print(len(train_4))
    print("\n")
    print('end')
    print(len(train_5))
    print("\n")

    print('\ntest')
    print(len(test_set))

    print(split_test_1)
    print(len(test_1))
    print("\n")
    print(split_test_2)
    print(len(test_2))
    print("\n")
    print(split_test_3)
    print(len(test_3))
    print("\n")
    print(split_test_4)
    print(len(test_4))
    print("\n")
    print('end')
    print(len(test_5))
    print("\n")

    return [[train_1, test_1], [train_2, test_2], [train_3, test_3], [train_4, test_4], [train_5, test_5]]


def split_5_uneven(train_set, test_set):
    # TODO: nicht ganz even wenn zahl nicht teilbar
    split_train_1 = math.floor(0.05 * len(train_set))
    split_train_2 = math.floor(0.10 * len(train_set)) + split_train_1
    split_train_3 = math.floor(0.20 * len(train_set)) + split_train_2
    split_train_4 = math.floor(0.25 * len(train_set)) + split_train_3

    print("heloooooooo" + str(split_train_1))
    print(split_train_2)
    print(split_train_3)
    print(split_train_4)

    split_test_1 = math.floor(0.05 * len(test_set))
    split_test_2 = math.floor(0.10 * len(test_set)) + split_test_1
    split_test_3 = math.floor(0.20 * len(test_set)) + split_test_2
    split_test_4 = math.floor(0.25 * len(test_set)) + split_test_3

    train_1 = train_set.iloc[0:split_train_1]
    train_2 = train_set.iloc[split_train_1:split_train_2]
    train_3 = train_set.iloc[split_train_2:split_train_3]
    train_4 = train_set.iloc[split_train_3:split_train_4]
    train_5 = train_set.iloc[split_train_4:, :]

    test_1 = test_set.iloc[0:split_test_1]
    test_2 = test_set.iloc[split_test_1:split_test_2]
    test_3 = test_set.iloc[split_test_2:split_test_3]
    test_4 = test_set.iloc[split_test_3:split_test_4]
    test_5 = test_set.iloc[split_test_4:, :]

    # test_1 = test_set.iloc[:split_test, :]
    # test_2 = test_set.iloc[split_test:, :]

    print('\n\ntrain')
    print(len(train_set))
    # print(len(train_1))
    # print(len(train_2))

    print(split_train_1)
    print(len(train_1))
    print("\n")
    print(split_train_2)
    print(len(train_2))
    print("\n")
    print(split_train_3)
    print(len(train_3))
    print("\n")
    print(split_train_4)
    print(len(train_4))
    print("\n")
    print('end')
    print(len(train_5))
    print("\n")

    print('\ntest')
    print(len(test_set))

    print(split_test_1)
    print(len(test_1))
    print("\n")
    print(split_test_2)
    print(len(test_2))
    print("\n")
    print(split_test_3)
    print(len(test_3))
    print("\n")
    print(split_test_4)
    print(len(test_4))
    print("\n")
    print('end')
    print(len(test_5))
    print("\n")

    return [[train_1, test_1], [train_2, test_2], [train_3, test_3], [train_4, test_4], [train_5, test_5]]


def create_config(input_train, input_test, duration_col, event_col, path):
    print(path)
    f = open(path + "/config.yml", "a")
    f.write("fc_rsf:\n\tfiles:\n\t\tinput: " + input_train + "\n\t\tinput_test: "
            + input_test + "\n\tparameters:" + "\n\t\tduration_col: " + duration_col + "\n\t\tevent_col: " + event_col)
    f.close()




import math
from functools import reduce
import numpy as np
import pandas as pd
from sas7bdat import SAS7BDAT
import sklearn
from sklearn.linear_model.tests.test_perceptron import random_state
from sklearn.preprocessing import OrdinalEncoder
#from sksurv.datasets import load_gbsg2, load_veterans_lung_cancer, load_whas500
from sksurv.metrics import concordance_index_censored
from sksurv.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
from sksurv.tree.tree import _array_to_step_function
from lifelines.datasets import load_rossi, load_gbsg2

from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
import os

"""
       This script parts a given data set into the input data for different clients, 
       with different distributions and adds the needed config file.
"""


def start_computation(path_root, dataset, duration_col, event_col, random_state_split, random_state_forest):
    path_root = path_root + "forest_state_" + str(random_state_forest)
    random_state_forest = str(random_state_forest)
    os.mkdir(path_root)
    train_set, test_set = split_test_set(dataset, random_state_split)

    test_set.to_csv(path_root + "/test_set.csv", index=False)
    train_set.to_csv(path_root + "/train_set.csv", index=False)

    compute_2_even(train_set, test_set, path_root, duration_col, event_col, random_state_forest)
    compute_2_uneven(train_set, test_set, path_root, duration_col, event_col, random_state_forest)
    compute_3_even(train_set, test_set, path_root, duration_col, event_col, random_state_forest)
    compute_3_uneven(train_set, test_set, path_root, duration_col, event_col, random_state_forest)
    compute_5_even(train_set, test_set, path_root, duration_col, event_col, random_state_forest)
    compute_5_uneven(train_set, test_set, path_root, duration_col, event_col, random_state_forest)
    #compute_10_even(train_set, test_set, path_root, duration_col, event_col, random_state_forest)
    #compute_10_uneven(train_set, test_set, path_root, duration_col, event_col, random_state_forest)


def compute_2_even(train_set, test_set, path, duration_col, event_col, random_state_forest):
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
        test_set.to_csv(current_dir_path + "/" + path_test, index=False)
        create_config(path_train, path_test, duration_col, event_col, current_dir_path, random_state_forest)

        current_client += 1


def compute_2_uneven(train_set, test_set, path, duration_col, event_col, random_state_forest):
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
        test_set.to_csv(current_dir_path + "/" + path_test, index=False)
        create_config(path_train, path_test, duration_col, event_col, current_dir_path, random_state_forest)

        current_client += 1


def compute_3_even(train_set, test_set, path, duration_col, event_col, random_state_forest):
    all_3_even_data = split_3_even(train_set, test_set)
    os.mkdir(path + "/3_even")
    current_client = 1
    for client_data in all_3_even_data:
        current_dir_path = path + "/3_even/client" + str(current_client)
        os.mkdir(current_dir_path)

        train_set = client_data[0]
        path_train = "train_" + str(current_client) + "_3_even.csv"
        train_set.to_csv(current_dir_path + "/" + path_train, index=False)

        test_set = client_data[1]
        path_test = "test_" + str(current_client) + "_3_even.csv"
        test_set.to_csv(current_dir_path + "/" + path_test, index=False)
        create_config(path_train, path_test, duration_col, event_col, current_dir_path, random_state_forest)

        current_client += 1


def compute_3_uneven(train_set, test_set, path, duration_col, event_col, random_state_forest):
    all_3_uneven_data = split_3_uneven(train_set, test_set)
    os.mkdir(path + "/3_uneven")
    current_client = 1
    for client_data in all_3_uneven_data:
        current_dir_path = path + "/3_uneven/client" + str(current_client)
        os.mkdir(current_dir_path)

        train_set = client_data[0]
        path_train = "train_" + str(current_client) + "_3_uneven.csv"
        train_set.to_csv(current_dir_path + "/" + path_train, index=False)

        test_set = client_data[1]
        path_test = "test_" + str(current_client) + "_3_uneven.csv"
        test_set.to_csv(current_dir_path + "/" + path_test, index=False)
        create_config(path_train, path_test, duration_col, event_col, current_dir_path, random_state_forest)

        current_client += 1



def compute_5_even(train_set, test_set, path, duration_col, event_col, random_state_forest):
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
        test_set.to_csv(current_dir_path + "/" + path_test, index=False)
        create_config(path_train, path_test, duration_col, event_col, current_dir_path, random_state_forest)

        current_client += 1


def compute_5_uneven(train_set, test_set, path, duration_col, event_col, random_state_forest):
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
        test_set.to_csv(current_dir_path + "/" + path_test, index=False)
        create_config(path_train, path_test, duration_col, event_col, current_dir_path, random_state_forest)

        current_client += 1


def compute_10_even(train_set, test_set, path, duration_col, event_col, random_state_forest):
    all_10_even_data = split_10_even(train_set, test_set)
    os.mkdir(path + "/10_even")
    current_client = 1
    for client_data in all_10_even_data:
        current_dir_path = path + "/10_even/client" + str(current_client)
        os.mkdir(current_dir_path)

        train_set = client_data[0]
        path_train = "train_" + str(current_client) + "_10_even.csv"
        train_set.to_csv(current_dir_path + "/" + path_train, index=False)

        test_set = client_data[1]
        path_test = "test_" + str(current_client) + "_10_even.csv"
        test_set.to_csv(current_dir_path + "/" + path_test, index=False)
        create_config(path_train, path_test, duration_col, event_col, current_dir_path, random_state_forest)

        current_client += 1


def compute_10_uneven(train_set, test_set, path, duration_col, event_col, random_state_forest):
    all_10_uneven_data = split_10_uneven(train_set, test_set)
    os.mkdir(path + "/10_uneven")
    current_client = 1
    for client_data in all_10_uneven_data:
        current_dir_path = path + "/10_uneven/client" + str(current_client)
        os.mkdir(current_dir_path)

        train_set = client_data[0]
        path_train = "train_" + str(current_client) + "_10_uneven.csv"
        train_set.to_csv(current_dir_path + "/" + path_train, index=False)

        test_set = client_data[1]
        path_test = "test_" + str(current_client) + "_10_uneven.csv"
        test_set.to_csv(current_dir_path + "/" + path_test, index=False)
        create_config(path_train, path_test, duration_col, event_col, current_dir_path, random_state_forest)

        current_client += 1



def split_test_set(dataset, state):
    train_set = dataset.sample(frac=0.75, random_state=state)
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


def split_3_even(train_set, test_set):
    split_train_1 = math.floor(len(train_set) / 3)
    split_train_2 = 2 * split_train_1


    split_test_1 = math.floor(len(test_set) / 5)
    split_test_2 = 2 * split_test_1


    train_1 = train_set.iloc[0:split_train_1]
    train_2 = train_set.iloc[split_train_1:split_train_2]
    train_3 = train_set.iloc[split_train_2:,:]


    test_1 = test_set.iloc[0:split_test_1]
    test_2 = test_set.iloc[split_test_1:split_test_2]
    test_3 = test_set.iloc[split_test_2: , :]


    return [[train_1, test_1], [train_2, test_2], [train_3, test_3]]

def split_3_uneven(train_set, test_set):
    split_train_1 = math.floor(0.2 * len(train_set))
    split_train_2 = math.floor(0.3 * len(train_set)) + split_train_1

    split_test_1 = math.floor(0.2 * len(test_set))
    split_test_2 = math.floor(0.3 * len(test_set)) + split_test_1


    train_1 = train_set.iloc[0:split_train_1]
    train_2 = train_set.iloc[split_train_1:split_train_2]
    train_3 = train_set.iloc[split_train_2:,:]


    test_1 = test_set.iloc[0:split_test_1]
    test_2 = test_set.iloc[split_test_1:split_test_2]
    test_3 = test_set.iloc[split_test_2: , :]


    return [[train_1, test_1], [train_2, test_2], [train_3, test_3]]


def split_5_even(train_set, test_set):
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


def split_10_even(train_set, test_set):
    split_train_1 = math.floor(len(train_set) / 10)
    split_train_2 = 2 * split_train_1
    split_train_3 = 3 * split_train_1
    split_train_4 = 4 * split_train_1
    split_train_5 = 5 * split_train_1
    split_train_6 = 6 * split_train_1
    split_train_7 = 7 * split_train_1
    split_train_8 = 8 * split_train_1
    split_train_9 = 9 * split_train_1


    split_test_1 = math.floor(len(test_set) / 10)
    split_test_2 = 2 * split_test_1
    split_test_3 = 3 * split_test_1
    split_test_4 = 4 * split_test_1
    split_test_5 = 5 * split_test_1
    split_test_6 = 6 * split_test_1
    split_test_7 = 7 * split_test_1
    split_test_8 = 8 * split_test_1
    split_test_9 = 9 * split_test_1


    train_1 = train_set.iloc[0:split_train_1]
    train_2 = train_set.iloc[split_train_1:split_train_2]
    train_3 = train_set.iloc[split_train_2:split_train_3]
    train_4 = train_set.iloc[split_train_3:split_train_4]
    train_5 = train_set.iloc[split_train_4:split_train_5]
    train_6 = train_set.iloc[split_train_5:split_train_6]
    train_7 = train_set.iloc[split_train_6:split_train_7]
    train_8 = train_set.iloc[split_train_7:split_train_8]
    train_9 = train_set.iloc[split_train_8:split_train_9]
    train_10 = train_set.iloc[split_train_9:, :]

    test_1 = test_set.iloc[0:split_test_1]
    test_2 = test_set.iloc[split_test_1:split_test_2]
    test_3 = test_set.iloc[split_test_2:split_test_3]
    test_4 = test_set.iloc[split_test_3:split_test_4]
    test_5 = test_set.iloc[split_test_4:split_test_5]
    test_6 = test_set.iloc[split_test_5:split_test_6]
    test_7 = test_set.iloc[split_test_6:split_test_7]
    test_8 = test_set.iloc[split_test_7:split_test_8]
    test_9 = test_set.iloc[split_test_8:split_test_9]
    test_10 = test_set.iloc[split_test_9:, :]

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
    print(split_train_5)
    print(len(train_5))
    print("\n")
    print(split_train_6)
    print(len(train_6))
    print("\n")
    print(split_train_7)
    print(len(train_7))
    print("\n")
    print(split_train_8)
    print(len(train_8))
    print("\n")
    print(split_train_9)
    print(len(train_9))
    print("\n")
    print('end')
    print(len(train_10))
    print("\n")

    result = [[train_1, test_1], [train_2, test_2], [train_3, test_3], [train_4, test_4], [train_5, test_5],
              [train_6, test_6], [train_7, test_7], [train_8, test_8], [train_9, test_9], [train_10, test_10]]

    return result

def split_10_uneven(train_set, test_set):
    split_train_1 = math.floor(0.02 * len(train_set))
    split_train_2 = math.floor(0.04 * len(train_set)) + split_train_1
    split_train_3 = math.floor(0.06 * len(train_set)) + split_train_2
    split_train_4 = math.floor(0.08 * len(train_set)) + split_train_3
    split_train_5 = math.floor(0.10 * len(train_set)) + split_train_4
    split_train_6 = math.floor(0.10 * len(train_set)) + split_train_5
    split_train_7 = math.floor(0.12 * len(train_set)) + split_train_6
    split_train_8 = math.floor(0.14 * len(train_set)) + split_train_7
    split_train_9 = math.floor(0.16 * len(train_set)) + split_train_8

    split_test_1 = math.floor(0.02 * len(test_set))
    split_test_2 = math.floor(0.04 * len(test_set)) + split_test_1
    split_test_3 = math.floor(0.06 * len(test_set)) + split_test_2
    split_test_4 = math.floor(0.08 * len(test_set)) + split_test_3
    split_test_5 = math.floor(0.10 * len(test_set)) + split_test_4
    split_test_6 = math.floor(0.10 * len(test_set)) + split_test_5
    split_test_7 = math.floor(0.12 * len(test_set)) + split_test_6
    split_test_8 = math.floor(0.14 * len(test_set)) + split_test_7
    split_test_9 = math.floor(0.16 * len(test_set)) + split_test_8


    train_1 = train_set.iloc[0:split_train_1]
    train_2 = train_set.iloc[split_train_1:split_train_2]
    train_3 = train_set.iloc[split_train_2:split_train_3]
    train_4 = train_set.iloc[split_train_3:split_train_4]
    train_5 = train_set.iloc[split_train_4:split_train_5]
    train_6 = train_set.iloc[split_train_5:split_train_6]
    train_7 = train_set.iloc[split_train_6:split_train_7]
    train_8 = train_set.iloc[split_train_7:split_train_8]
    train_9 = train_set.iloc[split_train_8:split_train_9]
    train_10 = train_set.iloc[split_train_9:, :]

    test_1 = test_set.iloc[0:split_test_1]
    test_2 = test_set.iloc[split_test_1:split_test_2]
    test_3 = test_set.iloc[split_test_2:split_test_3]
    test_4 = test_set.iloc[split_test_3:split_test_4]
    test_5 = test_set.iloc[split_test_4:split_test_5]
    test_6 = test_set.iloc[split_test_5:split_test_6]
    test_7 = test_set.iloc[split_test_6:split_test_7]
    test_8 = test_set.iloc[split_test_7:split_test_8]
    test_9 = test_set.iloc[split_test_8:split_test_9]
    test_10 = test_set.iloc[split_test_9:, :]

    # test_1 = test_set.iloc[:split_test, :]
    # test_2 = test_set.iloc[split_test:, :]

    print('\n\ntrain')
    print(len(test_set))
    # print(len(train_1))
    # print(len(train_2))

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
    print(split_test_5)
    print(len(test_5))
    print("\n")
    print(split_test_6)
    print(len(test_6))
    print("\n")
    print(split_test_7)
    print(len(test_7))
    print("\n")
    print(split_test_8)
    print(len(test_8))
    print("\n")
    print(split_test_9)
    print(len(test_9))
    print("\n")
    print('end')
    print(len(test_10))
    print("\n")

    result = [[train_1, test_1], [train_2, test_2], [train_3, test_3], [train_4, test_4], [train_5, test_5],
              [train_6, test_6], [train_7, test_7], [train_8, test_8], [train_9, test_9], [train_10, test_10]]

    return result


def create_config(input_train, input_test, duration_col, event_col, path, random_state):
    print(path)
    f = open(path + "/config.yml", "a")
    f.write("fc_rsf:\n  files:\n    input: " + input_train + "\n    input_test: "
            + input_test + "\n  parameters:" + "\n    duration_col: " + duration_col + "\n    event_col: " + event_col + "\n    random_state: " + random_state)
    f.close()



with SAS7BDAT('whas500.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
df.to_csv('whas500.csv')

whas = pd.read_csv('whas500.csv').drop('Unnamed: 0', axis=1).drop('ID', axis=1)

print(whas.head())



#print(gbsg2t.head())

os.mkdir('whas500')

i = 1


while i <= 3:
    j = 1
    os.mkdir('whas500/split_seed_' + str(i))
    while j <= 10:
        start_computation('whas500/split_seed_' + str(i) + '/', whas, 'LENFOL', 'FSTAT',random_state_split=i, random_state_forest=j)
        j += 1
    i += 1

#create_config("client1_whas500_2_even.csv", "client1_whas500_2_even_test.csv", "LENFOL", "FSTAT")


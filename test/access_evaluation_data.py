import re
from zipfile import ZipFile
import glob
import pandas as pd
import zipfile

from pandas import DataFrame


def all_scores_2(path, split_seed, name):
    i = 1
    all_information = []

    while i <= 10:
        zip_path = (glob.glob(path + str(i) + "/*.zip"))

        client_0 = None
        client_1 = None

        local_score = []

        if 'client_0' in zip_path[0]:
            client_0 = zip_path[0]
            client_1 = zip_path[1]

        else:
            client_0 = zip_path[1]
            client_1 = zip_path[0]

        all_information.append(get_list_out_of_zip(client_0, name, client_number=1, split_seed=split_seed, forest_seed=i))
        all_information.append(get_list_out_of_zip(client_1, name, client_number=2, split_seed=split_seed, forest_seed=i))
        i += 1

    return all_information

def all_scores_5(path, split_seed, name):
    i = 1
    all_information = []
    while i <= 10:
        zip_path = (glob.glob(path + str(i) + "/*.zip"))
        #print(zip_path)
        client_0 = None
        client_1 = None
        client_2 = None
        client_3 = None
        client_4 = None
        local_score = []
        j = 0
        while j<=4:
            if 'client_0' in zip_path[j]:
                client_0 = zip_path[j]
            if 'client_1' in zip_path[j]:
                client_1 = zip_path[j]
            if 'client_2' in zip_path[j]:
                client_2 = zip_path[j]
            if 'client_3' in zip_path[j]:
                client_3 = zip_path[j]
            if 'client_4' in zip_path[j]:
                client_4 = zip_path[j]
            j += 1
        all_information.append(
            get_list_out_of_zip(client_0, name, client_number=1, split_seed=split_seed, forest_seed=i))
        all_information.append(
            get_list_out_of_zip(client_1, name, client_number=2, split_seed=split_seed, forest_seed=i))
        all_information.append(
            get_list_out_of_zip(client_2, name, client_number=3, split_seed=split_seed, forest_seed=i))
        all_information.append(
            get_list_out_of_zip(client_3, name, client_number=4, split_seed=split_seed, forest_seed=i))
        all_information.append(
            get_list_out_of_zip(client_4, name, client_number=5, split_seed=split_seed, forest_seed=i))
        i += 1

    return all_information

def all_scores_3(path, split_seed, name):
    i = 1
    all_information = []
    while i <= 10:
        zip_path = (glob.glob(path + str(i) + "/*.zip"))
        #print(zip_path)
        client_0 = None
        client_1 = None
        client_2 = None
        client_3 = None
        client_4 = None
        local_score = []
        j = 0
        while j<=2:
            if 'client_0' in zip_path[j]:
                client_0 = zip_path[j]
            if 'client_1' in zip_path[j]:
                client_1 = zip_path[j]
            if 'client_2' in zip_path[j]:
                client_2 = zip_path[j]
            j += 1
        all_information.append(
            get_list_out_of_zip(client_0, name, client_number=1, split_seed=split_seed, forest_seed=i))
        all_information.append(
            get_list_out_of_zip(client_1, name, client_number=2, split_seed=split_seed, forest_seed=i))
        all_information.append(
            get_list_out_of_zip(client_2, name, client_number=3, split_seed=split_seed, forest_seed=i))
        i += 1

    return all_information

def get_list_out_of_zip(path_to_zip, name, client_number, split_seed, forest_seed):
    #print(path_to_zip)
    zf_0 = zipfile.ZipFile(path_to_zip)
    df_0 = pd.read_csv(zf_0.open('evaluation_result.csv'))
    #print(df_0.head())
    df_0_list = df_0.values.tolist()[0]
    #print(df_0_list)
    df_0_list.insert(0, name)
    df_0_list.insert(1, client_number)
    df_0_list.insert(2, str(split_seed))
    df_0_list.insert(3, str(forest_seed))
    print(df_0_list)
    return df_0_list

even_2 = all_scores_2("rossi_1/even_2_state1/", 1, "2_even")
uneven_2 = all_scores_2("rossi_1/uneven_2_state1/", 1, "2_uneven")
even_3 = all_scores_3("rossi_1/even_3_state1/", 1, "3_even")
uneven_3 = all_scores_3("rossi_1/uneven_3_state1/", 1, "3_uneven")
even_5 = all_scores_5("rossi_1/even_5_state1/", 1, "5_even")
uneven_5 = all_scores_5("rossi_1/uneven_5_state1/", 1, "5_uneven")

all_data = even_2 + uneven_2 + even_3 + uneven_3+ even_5 + uneven_5


df = DataFrame(all_data,columns=['Name','client','split_seed', 'forest_seed', 'cindex_on_global_model', 'global_c_index_mean', 'global_c_index_weighted', 'training_samples', 'test_samples', 'concordant_pairs'])

print(df.head())

df.to_csv('test.csv')

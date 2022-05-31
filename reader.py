import random
import numpy as np

from sklearn.model_selection import train_test_split

from utils import get_requests_from_file,batch_generator,one_by_one_generator
from vocab import Vocabulary


def process_request(req,vocab):
    """
    Splits a request into lines and convert a string into ints.
    """
    seq = vocab().string_to_int(text=req)
    l = len(seq)

    return seq, l


def read(data_path):

    vocab = Vocabulary
    data = get_requests_from_file(data_path)
    print("Downloaded {} samples".format(len(data)))
    count=0
    for i in range(0, len(data)):
        if(len(process_request(data[i],vocab)[0])>1200):
            count=count+1
    print(count,"------")

    # Eliminate wrong HTTP requests.
    data_eliminated=[]
    for i in range(0, len(data)):
        if(len(process_request(data[i],vocab)[0])>1200):
            continue
        data_eliminated.append(data[i])
    print(len(data_eliminated),"data_eliminated length")

    data_array = np.zeros((len(data_eliminated),1200),dtype=int)
    for i in range(0,len(data_eliminated)):#Investigate 789.
        data_array[i,:len(process_request(data_eliminated[i],vocab)[0])] = process_request(data_eliminated[i],vocab)[0]

    return data_array




# train_data_all = read("datasets/vulnbank_train.txt")
# print(train_data_all,"train_data_all")
# print(train_data_all.shape,"train_data_all.shape")
# print("....")


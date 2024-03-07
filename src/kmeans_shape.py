from sklearn.cluster import KMeans
import numpy as np
import argparse
import math
import time
import os

def cluster_distance(x_1,x_all):
    list = []
    for i in range (0,x_all.shape[0]):
        hamming_distance = x_1.shape[0] - np.sum(x_1 == x_all[i,:])
        list.append(hamming_distance)
    sorted_id = sorted(range(len(list)),key=lambda k: list[k],reverse=True)
    return sorted_id,list

def eu_distance(x_1,x_all):
    list = []
    for i in range (0,x_all.shape[0]):
        eu_distance = np.linalg.norm(x_1 - x_all[i,:])
        list.append(eu_distance)
    sorted_id = sorted(range(len(list)),key=lambda k: list[k],reverse=True)
    return sorted_id,list

def load_raw_ts(path, dataset):
    path = path + dataset + "/"
    x_train = np.load(path + 'X_train.npy')
    y_train = np.load(path + 'y_train.npy')
    x_test = np.load(path + 'X_test.npy')
    y_test = np.load(path + 'y_test.npy')
    x_train = np.transpose(x_train, axes=(0, 2, 1))
    x_test = np.transpose(x_test, axes=(0, 2, 1))

    nclass = int(np.amax(y_train)) + 1
    dim_length = x_train.shape[2]
    dim_v = x_train.shape[1]

    return x_train, x_test, y_train, y_test, dim_length, dim_v, nclass

def slide_TS(window_size, X):
    '''
    tensor version
    slide the multivariate time series tensor from 3D to 2D
    add the dimension label to each variate
    add step to reduce the num of new TS
    '''
    dim_length = X.shape[1]
    X_alpha = X[:, 0 : window_size]

    # determine step
    if (dim_length <= 50) :
        step = 1
    elif (dim_length > 50 and dim_length <= 100):
        step = 2
    elif (dim_length > 100 and dim_length <= 300):
        step = 3
    elif (dim_length > 300 and dim_length <= 1000):
        step = 4
    elif (dim_length > 1000 and dim_length <= 1500):
        step = 5
    elif (dim_length > 1500 and dim_length <= 2000):
        step = 7
    elif (dim_length > 2000 and dim_length <= 3000):
        step = 10
    else:
        step = 1000

    # determine step number
    step_num = int(math.ceil((dim_length -window_size)/step))

    # still slide to 2D
    for k in range(1, dim_length-window_size+1, step):

        X_temp = X[:, k : window_size + k]
        X_alpha = np.concatenate((X_alpha, X_temp), axis = 1)

    # numpy reshape (number of instances, windowsize * number of subsequence ) to (number of instances* number of subsequence, windowsize)
    X_beta = np.reshape(X_alpha,( (X_alpha.shape[0]) * (step_num+1), window_size))
    return X_beta

def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')


def cluster_variable_window(data, window_size, dim_v, num_center_window) :
    number_instance = data.shape[0]
    shapes_list = []
    shapes_distance_list = []
    shapes_cluster_list = []
    for v in range(0,dim_v):
        train_data_v = data[:, v, :]
        shape_all_window = np.empty([number_instance,0,window_size[2]])
        shape_all_window_distance = np.empty([number_instance, 0])
        shape_all_window_cluster = np.empty([number_instance, 0])
        for i in range(0,3):
            shape_one_window = []
            shape_one_window_distance = []
            shape_one_window_cluster = []
            x_subsequence = slide_TS(window_size[i],train_data_v)

            kmeans = KMeans(n_clusters=num_center_window[i]).fit(x_subsequence)
            all_distance = kmeans.fit_transform(x_subsequence)
            subsequences_labels = kmeans.labels_

            subsequence_per_instance = int(x_subsequence.shape[0]/train_data_v.shape[0])
            #cluster_centers_temp = kmeans.cluster_centers_
            for j in range(0,number_instance):
                one_instance_distance = all_distance[subsequence_per_instance * j: subsequence_per_instance * (j+1),:]
                one_instance_label = subsequences_labels[subsequence_per_instance * j: subsequence_per_instance * (j+1)]
                one_min_key = np.argmin(one_instance_distance,axis=0)
                one_min = np.min(one_instance_distance,axis=0)
                for k in range(0,one_min_key.shape[0]):
                    shape_temp = x_subsequence[one_min_key[k]+subsequence_per_instance * j,:]
                    shape_one_window.append(padarray(shape_temp,window_size[2]))
                    shape_one_window_distance.append(one_min[k])
                    shape_one_window_cluster.append(one_instance_label[one_min_key[k]])
                    #if k==0 and j==15 and v==5 and i==1:
                    #   out_2 = shape_temp  for yanzheng

            shape_one_window_all = np.array(shape_one_window).reshape(number_instance,num_center_window[i],window_size[2])
            shape_one_window_all_distance = np.array(shape_one_window_distance).reshape(number_instance,num_center_window[i])
            shape_one_window_all_cluster = np.array(shape_one_window_cluster).reshape(number_instance,num_center_window[i])

            shape_all_window = np.concatenate((shape_all_window,shape_one_window_all), axis=1)
            shape_all_window_distance = np.concatenate((shape_all_window_distance, shape_one_window_all_distance), axis=1)
            shape_all_window_cluster = np.concatenate((shape_all_window_cluster, shape_one_window_all_cluster),
                                                       axis=1)

        shapes_list.append(shape_all_window)
        shapes_distance_list.append(shape_all_window_distance)
        shapes_cluster_list.append(shape_all_window_cluster)
    shapes_all = np.array(shapes_list)
    shapes_all_distance = np.array(shapes_distance_list)
    shapes_all_cluster = np.array(shapes_cluster_list)
    shapes = np.transpose(shapes_all, axes=(1, 0, 2, 3))
    distance = np.transpose(shapes_all_distance, axes=(1,0,2))
    clusters = np.transpose(shapes_all_cluster, axes=(1,0,2))
    shapes = shapes.reshape(number_instance, dim_v * sum(num_center_window), window_size[2])
    distance = distance.reshape(number_instance, dim_v * sum(num_center_window))
    clusters = clusters.reshape(number_instance, dim_v * sum(num_center_window))

    return shapes, distance,clusters



def get_window(l):
    if l <= 200:
        window_size = [int(0.1 * l), int(0.2 * l), int(0.3 * l)]
    elif ( l>200 and l<=500) :
        window_size = [int(0.05 * l), int(0.1 * l), int(0.2 * l)]
    elif ( l>500 and l <= 1000):
        window_size = [int(0.05 * l), int(0.1 * l), int(0.2 * l)]
    elif (l > 1000):
        window_size = [50, 100, 200]
    return window_size
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="NATOPS", #NATOPS
                    help='time series dataset. Options: See the datasets list')
parser.add_argument('--number_shapes', type=int, default="1024", #NATOPS
                    help='time series dataset. Options: See the datasets list')
parser.add_argument('--data_path', type=str, default="../data/raw/", #NATOPS
                    help='time series dataset. Options: See the datasets list')

args = parser.parse_args()
data_path = args.data_path
dataset = args.dataset
number_shapes = args.number_shapes

train_data, test_data, train_label, test_label, dim_length, dim_v, nclass = load_raw_ts(data_path, dataset)

window_size = get_window(dim_length)


num_center_window = [int((3*number_shapes)/(6*dim_v)), int((2*number_shapes)/(6*dim_v)), int((1*number_shapes)/(6*dim_v))]
tol_shapes = dim_v * sum(num_center_window)

data_all = np.concatenate((train_data,test_data), axis= 0)

a1 = time.perf_counter()

shapes, distance,clusters = cluster_variable_window(data_all, window_size, dim_v, num_center_window)

a2 = time.perf_counter()

train_shapes = shapes[0:train_data.shape[0],:,:]
train_distance = distance[0:train_data.shape[0],:]
train_clusters = clusters[0:train_data.shape[0],:]

test_shapes = shapes[train_data.shape[0]:train_data.shape[0] + test_data.shape[0],:,:]
test_distance = distance[train_data.shape[0]:train_data.shape[0] + test_data.shape[0],:]
test_clusters = clusters[train_data.shape[0]:train_data.shape[0] + test_data.shape[0],:]

path = '../data/preprocess/' + dataset + '/' + str(number_shapes)
os.makedirs(path)
np.save(path + '/train',train_shapes)
np.save(path + '/test',test_shapes)

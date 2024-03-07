import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="NATOPS", #NATOPS
                    help='time series dataset. Options: See the datasets list')
parser.add_argument('--data_path', type=str, default="../data/raw/", #NATOPS
                    help='time series dataset. Options: See the datasets list')
args = parser.parse_args()

dataset = args.dataset
data_path = args.data_path
number_shapes= [128,256,512,1024,2048,4096]
for item in number_shapes:
    cmd = 'python kmeans_shape.py --dataset '+ dataset + ' --number_shapes ' + str(int(item)) + ' --data_path '+ data_path
    os.system(cmd)
    print('finished window size ' + str(item))
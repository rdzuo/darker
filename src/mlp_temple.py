import numpy as np
import torch.utils.data as Data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softmax
import os
import argparse


class MLP_para(nn.Module):
    def __init__(self,d_length,d_mlp):
        super(MLP_para, self).__init__()
        self.d_length = d_length
        self.d_mlp = d_mlp
        self.number_shapes = number_shapes
        self.layers_1 = nn.Sequential(
            nn.Linear(d_length, d_mlp[0]),
            nn.ReLU(),
            nn.Linear(d_mlp[0], d_length)
        )
        self.layers_2 = nn.Sequential(
            nn.Linear(d_length, d_mlp[1]),
            nn.ReLU(),
            nn.Linear(d_mlp[1], d_length)
        )
        self.layers_3 = nn.Sequential(
            nn.Linear(number_shapes, d_mlp[2]),
            nn.ReLU(),
            nn.Linear(d_mlp[2], number_shapes)
        )
        self.layer_norm = nn.LayerNorm(d_length,d_length)

    def forward(self, x,w,x_prime,need_x = False):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = self.layers_1(x)
        w = self.layers_2(w)
        w = self.layer_norm(w)
        x_prime = self.layers_3(x_prime)
        y = torch.bmm(x, w)
        #output = self.layer_norm(torch.bmm(y, x_prime))
        output = torch.bmm(y,x_prime)
        if need_x:
            return x, x_prime
        else:
            return output



#def generate_one_dim_gaussn(n_w,):


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="BasicMotions", #NATOPS
                    help='time series dataset. Options: See the datasets list')
parser.add_argument('--number_shapes', type=str, default="4096", #NATOPS
                    help='time series dataset. Options: See the datasets list')
parser.add_argument('--shape_path', type=str, default="../data/preprocess/", #NATOPS
                    help='time series dataset. Options: See the datasets list')

args = parser.parse_args()


dataset = args.dataset
shape_path =args.shape_path
number_shapes_name = args.number_shapes



# number of random weights
n_w = 200

# d model  d_projection (optional)
d_model = 16
d_mlp = [64,16,64]
d_projection = 32
epochs = 50



# load shape
train_shapes = np.load(shape_path + dataset + "/" + number_shapes_name+'/'+"train.npy")
test_shapes = np.load(shape_path + dataset + "/" + number_shapes_name+ '/'+"test.npy")

all_shapes = np.concatenate((train_shapes,test_shapes),axis=0)
#all_shapes = np.random.rand(80,504,30) * 10
all_instance = all_shapes.shape[0]

number_instance_train = train_shapes.shape[0]
number_instance_test = test_shapes.shape[0]

# d_shape_length
d_shape_length = train_shapes.shape[2]
number_shapes = train_shapes.shape[1]

# prepare random weights for training

# simplest method: one dim gauss
# normal distribution w vector with mean = 0, stand = 5
w_initail = np.random.normal(0,5,n_w * d_shape_length * d_shape_length)
w_all = np.reshape(w_initail,(n_w,d_shape_length,d_shape_length))
# to torch vision
w_all = torch.FloatTensor(w_all)

# device
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
seed = 42
#np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

loss_module = nn.MSELoss()

train_loss_all_instance = []

list_fai_x =[]
list_fai_x_prime = []
list_w_1_weights= []
list_w_1_bias = []
list_w_2_weights = []
list_w_2_bias = []



for i in range (0,all_instance):
    shape_tensor = torch.FloatTensor(all_shapes[i,:,:])
    shape_tensor_T = torch.FloatTensor(all_shapes[i,:,:].T)
    shape_for_learn = shape_tensor.expand(n_w, number_shapes, d_shape_length)
    shape_for_learn_T = shape_tensor_T.expand(n_w, d_shape_length, number_shapes)
    # get labels
    middle_one = torch.bmm(shape_for_learn, w_all)
    middle_two = torch.bmm(middle_one, shape_for_learn_T)
    labels = softmax(middle_two, dim=-1)
    # prepare data loader
    train_dataset = Data.TensorDataset(shape_for_learn, w_all, shape_for_learn_T, labels)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    '''
        a = shape_for_learn[0, :, :].expand(1, number_shapes, d_shape_length)
    val_dataset = Data.TensorDataset(shape_for_learn[0, :, :].expand(1, number_shapes, d_shape_length), 
                                     w_all[0, :, :], shape_for_learn_T[0, :, :], labels)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)
    '''

    val_dataset = Data.TensorDataset(shape_for_learn, w_all, shape_for_learn_T, labels)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)

    # model
    model = MLP_para(d_shape_length, d_mlp)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    train_loss_one_instance = []
    for epoch in range(epochs):
        model.train()
        train_losses = []

        for j, (x_1, w_1, x_prime_1, labels_1) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(x_1.to(device), w_1.to(device), x_prime_1.to(device))
            loss = loss_module(outputs, labels_1.to(device))
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        print('epoch : {}, train loss : {:}'\
              .format(epoch + 1, np.mean(train_losses)))

        train_loss_one_instance.append((np.mean(train_losses)).item())
    train_loss_one_instance_array = np.array(train_loss_one_instance)
    train_loss_all_instance.append(train_loss_one_instance_array)
    # store fai_x, fai_x_prime
    model.eval()
    with torch.no_grad():
        for k, (x_2, w_2, x_prime_2, labels_2) in enumerate(val_loader):
            optimizer.zero_grad()
            if k > 0:
                break
            fai_x, fai_x_prime = model(x_2.to(device), w_2.to(device), x_prime_2.to(device),need_x = True)


    list_fai_x.append(np.squeeze((fai_x.cpu()).numpy(), axis=0))
    list_fai_x_prime.append(np.squeeze((fai_x_prime.cpu()).numpy(), axis=0 ))


    # store weights for w
    parameters = model.state_dict()
    w_1_weights_temp = (parameters['layers_2.0.weight'].cpu()).numpy()
    w_1_bias_temp = (parameters['layers_2.0.bias'].cpu()).numpy()
    w_2_weights_temp = (parameters['layers_2.2.weight'].cpu()).numpy()
    w_2_bias_temp = (parameters['layers_2.2.bias'].cpu()).numpy()

    list_w_1_weights.append(w_1_weights_temp)
    list_w_1_bias.append(w_1_bias_temp)
    list_w_2_weights.append(w_2_weights_temp)
    list_w_2_bias.append(w_2_bias_temp)

    del model
    print('finished instance ' + str(i))

fai_x_all = np.array(list_fai_x)
fai_x_prime_all= np.array(list_fai_x_prime)
w_1_weights_all = np.array(list_w_1_weights)
w_1_bias_all =np.array(list_w_1_bias)
w_2_weights_all = np.array(list_w_2_weights)
w_2_bias_all = np.array(list_w_2_bias)


loss_all_array = np.array(train_loss_all_instance).reshape(all_instance, epochs)






path = '../data/preprocess_learned/' + dataset + '/' + number_shapes_name
os.makedirs(path)

np.savez(path + '/learned_parameters',fai_x_all,fai_x_prime_all,w_1_weights_all,w_1_bias_all,w_2_weights_all,w_2_bias_all)






# prepare data loader

# save fai(x)

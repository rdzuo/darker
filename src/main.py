from tqdm import tqdm
import time
import numpy as np
from models import analyze_classification, NoFussCrossEntropyLoss, \
\
    Transformer_learned_base,Transformer_learned_learn, Transformer_learned_performer, Transformer_learned_trig, ProbAttention
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
import torch
import logging
import argparse


logging.basicConfig(filename='new.log', filemode='w', format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

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


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="BasicMotions", #NATOPS
                    help='time series dataset. Options: See the datasets list')

parser.add_argument('--number_shapes_name', type=str, default="4096", #NATOPS
                    help='time series dataset. Options: See the datasets list')

parser.add_argument('--epochs', type=int, default="100", #NATOPS
                    help='time series dataset. Options: See the datasets list')

parser.add_argument('--batch_size', type=int, default="32", #NATOPS
                    help='time series dataset. Options: See the datasets list')

parser.add_argument('--model_name', type=str, default="learned", #NATOPS
                    help='time series dataset. Options: See the datasets list')

parser.add_argument('----learn_rate', type=float, default="0.001", #NATOPS
                    help='time series dataset. Options: See the datasets list')

parser.add_argument('--data_path', type=str, default="../data/raw/", #NATOPS
                    help='time series dataset. Options: See the datasets list')

parser.add_argument('--shape_path', type=str, default="../data/preprocess/", #NATOPS
                    help='time series dataset. Options: See the datasets list')

parser.add_argument('--parameter_path', type=str, default="../data/preprocess_learned/", #NATOPS
                    help='time series dataset. Options: See the datasets list')


args = parser.parse_args()






dataset = args.dataset
number_shapes_name = args.number_shapes_name
model_name = args.model_name


epochs = args.epochs
batch_size = args.batch_size

data_path = args.data_path
shape_path = args.shape_path
parameter_path = args.parameter_path


# d model  d_projection (optional)
d_model = 16
d_mlp = [64,16,64]
d_projection = 32


# load data
train_data, test_data, train_label, test_label, dim_length, dim_v, nclass = load_raw_ts(data_path, dataset)

# load shape
train_shapes = np.load(shape_path + dataset + "/" + number_shapes_name+'/'+"train.npy")
test_shapes = np.load(shape_path + dataset + "/" + number_shapes_name+ '/'+"test.npy")

number_shape = train_shapes.shape[1] # number of shapes per instance
d_length = train_shapes.shape[2] # length of shape after padding


train_number_instance = train_data.shape[0]
test_number_instance = test_data.shape[0]
totol_instance = train_number_instance + test_number_instance

# load parameters
parameters = np.load(parameter_path + dataset + "/" + number_shapes_name + '/' +"learned_parameters.npz")
fai_x_all = parameters["arr_0"]
fai_x_prime_all = parameters["arr_1"]
w_1_weights_all = parameters["arr_2"]
w_1_bias_all = np.expand_dims(parameters["arr_3"],axis=1)
w_2_weights_all = parameters["arr_4"]
w_2_bias_all = np.expand_dims(parameters["arr_5"],axis=1)

train_dataset = Data.TensorDataset(torch.FloatTensor(train_shapes), torch.FloatTensor(fai_x_all[0:train_number_instance,:,:]),
                                   torch.FloatTensor(fai_x_prime_all[0:train_number_instance,:,:]),
                                   torch.transpose((torch.FloatTensor(w_1_weights_all[0:train_number_instance,:,:])),1,2),
                                   torch.FloatTensor(w_1_bias_all[0:train_number_instance,:]).expand(train_number_instance,d_length,d_mlp[1]),
                                   torch.transpose((torch.FloatTensor(w_2_weights_all[0:train_number_instance,:,:])),1,2),
                                   torch.FloatTensor(w_2_bias_all[0:train_number_instance,:]).expand(train_number_instance,d_length,d_length),
                                   torch.FloatTensor(train_label))

test_dataset = Data.TensorDataset(torch.FloatTensor(test_shapes), torch.FloatTensor(fai_x_all[train_number_instance:totol_instance,:,:]),
                                  torch.FloatTensor(fai_x_prime_all[train_number_instance:totol_instance,:,:]),
                                   torch.transpose((torch.FloatTensor(w_1_weights_all[train_number_instance:totol_instance,:,:])),1,2),
                                  torch.FloatTensor(w_1_bias_all[train_number_instance:totol_instance,:]).expand(test_number_instance,d_length,d_mlp[1]),
                                   torch.transpose((torch.FloatTensor(w_2_weights_all[train_number_instance:totol_instance,:,:])),1,2),
                                  torch.FloatTensor(w_2_bias_all[train_number_instance:totol_instance,:]).expand(test_number_instance,d_length,d_length),
                                  torch.FloatTensor(test_label))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader =DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


'''

'''
# device
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
#device = torch.device('cpu')
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



m = 128
'''

model_1 = Transformer_learned_learn(d_length,d_model,number_shape,nclass)
model_2 = Transformer_learned_base(d_length,d_model,number_shape,nclass)
model_3 = Transformer_learned_performer(d_length,d_model,number_shape,nclass,m)

model_1.to(device)
model_2.to(device)
model_3.to(device)

for i_test, a_test in enumerate(train_loader):
    X_test, fai_x_test, fai_x_prime_test, w_1_test, b_1_test, w_2_test, b_2_test, label_test = a_test
    if i_test == 0:
        break
flops_1, params_1 = profile(model_1, inputs=(X_test.to(device), fai_x_test.to(device), fai_x_prime_test.to(device),
                                         w_1_test.to(device), b_1_test.to(device), w_2_test.to(device), b_2_test.to(device),))
flops_2, params_2 = profile(model_2, inputs=(X_test.to(device), fai_x_test.to(device), fai_x_prime_test.to(device),
                                         w_1_test.to(device), b_1_test.to(device), w_2_test.to(device), b_2_test.to(device),))
flops_3, params_3 = profile(model_3, inputs=(X_test.to(device), fai_x_test.to(device), fai_x_prime_test.to(device),
                                         w_1_test.to(device), b_1_test.to(device), w_2_test.to(device), b_2_test.to(device),))

'''

# to calculate the Flops and parameters





if model_name == 'learned':
    model = Transformer_learned_learn(d_length, d_model, number_shape, nclass)
elif model_name == 'positive':
    model = Transformer_learned_performer(d_length, d_model, number_shape, nclass, m)
elif model_name == 'trig':
    model = Transformer_learned_trig(d_length, d_model, number_shape, nclass,m)
elif model_name == 'full':
    model = Transformer_learned_base(d_length, d_model, number_shape, nclass)
elif model_name == 'informer':
    model = ProbAttention(d_length, d_model, number_shape, nclass)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


loss_module = NoFussCrossEntropyLoss(reduction='none')



train_time = 0
for epoch in tqdm(range(1, epochs + 1), desc='Training Epoch', leave=False):
    epoch_start_time = time.time()
    model.train()
    total_samples = 0
    epoch_loss = 0
    for i, a in enumerate(train_loader):
        X, fai_x, fai_x_prime, w_1, b_1, w_2, b_2, label = a
        label = label.to(device)
        prediction = model(X.to(device), fai_x.to(device), fai_x_prime.to(device),
                           w_1.to(device), b_1.to(device), w_2.to(device), b_2.to(device))
        loss = loss_module(prediction, label)
        batch_loss = torch.sum(loss)
        mean_loss = batch_loss / len(loss)
        optimizer.zero_grad()
        mean_loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        optimizer.step()

        total_samples += len(loss)
        epoch_loss += batch_loss.item()

    epoch_runtime = time.time() - epoch_start_time
    train_time = train_time + epoch_runtime
    epoch_loss = epoch_loss / total_samples
    logger.info("Epoch runtime: {} seconds\n".format(epoch_runtime))
    print('epoch : {}, train loss : {:.4f}' \
            .format(epoch, epoch_loss))
    if epoch == epochs or (epoch % 2 == 0):
        logger.info("Evaluating on validation set ...")
        eval_start_time = time.time()
        model.eval()
        with torch.no_grad():
            epoch_loss = 0  # total loss of epoch
            total_samples = 0

            per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
            for i, a in enumerate(test_loader):
                X, fai_x, fai_x_prime, w_1, b_1, w_2, b_2, label = a
                label = label.to(device)
                prediction = model(X.to(device), fai_x.to(device), fai_x_prime.to(device),
                                   w_1.to(device), b_1.to(device), w_2.to(device), b_2.to(device))
                loss = loss_module(prediction, label)
                batch_loss = torch.sum(loss).cpu().item()
                mean_loss = batch_loss / len(loss)  # mean loss (over samples)

                per_batch['targets'].append(label.cpu().numpy())
                per_batch['predictions'].append(prediction.cpu().numpy())
                per_batch['metrics'].append([loss.cpu().numpy()])

            eval_time = time.time() - eval_start_time

            prediction = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
            probs = torch.nn.functional.softmax(prediction)
            prediction = torch.argmax(probs, dim=1).cpu().numpy()
            probs = probs.cpu().numpy()
            targets = np.concatenate(per_batch['targets'], axis=0).flatten()
            class_names = np.arange(probs.shape[1])
            accuracy = analyze_classification(prediction, targets, class_names)

result = dataset + ' ' + model_name + ' batch_size ' + str(batch_size) + ' number_shapes ' + number_shapes_name + ' accuracy ' + str(accuracy) + ' train_time ' + str(train_time) + "\n"
print(result)
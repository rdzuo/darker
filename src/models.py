from torch import nn
import torch
from torch.nn.functional import linear, softmax, dropout, pad
from torch.nn import functional as F
from sklearn import metrics
import numpy as np
from math import sqrt

'''
def choose_model(model_tobuild):
    if model_tobuild = 'transformerbase':
        model = Transformer_base()
'''



def analyze_classification(y_pred, y_true, class_names):
    maxcharlength = 35

    in_pred_labels = set(list(y_pred))
    y_true = y_true.astype(int)
    in_true_labels = set(list(y_true))

    existing_class_ind = sorted(list(in_pred_labels | in_true_labels))
    class_strings = [str(name) for name in class_names]  # needed in case `class_names` elements are not strings
    existing_class_names = [class_strings[ind][:min(maxcharlength, len(class_strings[ind]))] for ind in
                                 existing_class_ind]  # a little inefficient but inconsequential

    # Confusion matrix
    ConfMatrix = metrics.confusion_matrix(y_true, y_pred)


    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    ConfMatrix_normalized_row = ConfMatrix.astype('float') / ConfMatrix.sum(axis=1)[:, np.newaxis]

    # Analyze results
    total_accuracy = np.trace(ConfMatrix) / len(y_true)
    print('Overall accuracy: {:.3f}\n'.format(total_accuracy))
    return total_accuracy


def iid_gaussian(m, d):
    return torch.randn((m, d))

def orthogonal_gaussian(m, d):
    def orthogonal_square():
        q, _ = torch.qr(iid_gaussian(d, d))
        return q.t()

    num_squares = int(m / d)
    blocks = [orthogonal_square() for _ in range(num_squares)]

    remainder = m - d * num_squares
    if remainder:
        blocks.append(orthogonal_square()[:remainder])

    matrix = torch.vstack(blocks)
    matrix /= torch.sqrt(torch.tensor(num_squares + remainder / d))

    return matrix


def phi_trig(h,m,random_feats):
    sin = lambda x: torch.sin(2 * np.pi * x)
    cos = lambda x: torch.cos(2 * np.pi * x)
    fs = [sin, cos]


    def func(x):
        return (h(x)/torch.sqrt(torch.tensor(m)) *
                torch.cat([f(torch.einsum("bld,bmd->blm", x, random_feats)) for f in fs], dim=-1)
        )

    return func


def phi_postive(h,m,random_feats):
    def func(x):
        return (h(x)/torch.sqrt(torch.tensor(m)) *
                torch.cat([torch.exp(torch.einsum("bld,bmd->blm", x, random_feats))], dim=-1)
        )

    return func

def mse(a, b):
    return torch.square(a - b).mean()



class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(1), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)





class Transformer_learned_learn(nn.Module):
    def __init__(self, d_length,d_model,number_shape,number_class, device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')):
        super(Transformer_learned_learn, self).__init__()
        self.d_length = d_length
        self.d_model = d_model
        self.number_shape = number_shape
        self.number_class = number_class
        self.relu = nn.ReLU()
        self.layer_q = nn.Linear(d_length,d_model,bias=False)
        self.layer_k = nn.Linear(d_length, d_model, bias=False)
        self.layer_v = nn.Linear(d_length,d_model,bias=False)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_length, d_length)
        self.act = F.gelu
        self.layer_final = nn.Linear(d_model * number_shape, number_class)
        self.one = torch.unsqueeze(torch.eye(d_length),0).to(device)

        self.layer_add = nn.Linear(d_length, d_model)
        self.dropout= nn.Dropout(0.1)
        self.pe = nn.Parameter(torch.empty(1, number_shape, d_length))
        nn.init.uniform_(self.pe, -0.02, 0.02)
    def forward(self,input_embedding,fai_x, fai_x_prime,
                           w_1, b_1, w_2, b_2):

        value = self.layer_v(input_embedding)
        #output = torch.bmm(fai_x_prime,value)

        # Attn = X * (W_Q * W_K.T) * X.T * V
        # Attn_learn = fai(X) * fai(W_Q * W_K.T) * fai(X.T) * V

        # get W_Q from the layer Q
        w_q = self.layer_q(((self.one).expand(input_embedding.shape[0],self.d_length,self.d_length)))
        # get W_K from the layer K
        w_k = self.layer_k(((self.one).expand(input_embedding.shape[0],self.d_length,self.d_length)))
        # W_in = W_Q * W_K.T
        w_in = torch.bmm(w_q,torch.transpose(w_k,1,2))



        w_out = torch.bmm(w_in, w_1) + b_1
        w_out = self.relu(w_out)
        w_out = torch.bmm(w_out, w_2) + b_2
        w_out = self.layer_norm(w_out)

        # output = fai(X.T) * V
        output = torch.bmm(fai_x_prime,value)
        output = torch.bmm(w_out, output)
        output = torch.bmm(fai_x, output)


        output = output + self.layer_add(input_embedding)

        output = self.dropout(output)

        output = output.reshape(output.shape[0], -1)
        output = self.layer_final(output)

        return output


class Transformer_learned_base(nn.Module):
    def __init__(self, d_length,d_model,number_shape,number_class, device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')):
        super(Transformer_learned_base, self).__init__()
        self.d_length = d_length
        self.d_model = d_model
        self.number_shape = number_shape
        self.number_class = number_class
        self.relu = nn.ReLU()
        #self.layer_q = nn.Linear(d_length,d_model,bias=False)
        #self.layer_k = nn.Linear(d_length,d_model,bias=False)
        #w_q = torch.ones((d_length,d_model), requires_grad= True).to(device)
        #self.w_q = torch.nn.parameter(w_q)
        #w_k = torch.ones((d_length, d_length), requires_grad=True).to(device)
        #self.w_k = Parameter(w_k)
        self.layer_q = nn.Linear(d_length,d_model,bias=False)
        self.layer_k = nn.Linear(d_length, d_model, bias=False)
        self.layer_v = nn.Linear(d_length,d_model,bias=False)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_length, d_length)
        self.act = F.gelu
        self.layer_final = nn.Linear(d_model * number_shape, number_class)
        self.one = torch.unsqueeze(torch.eye(d_length),0).to(device)

        self.layer_add = nn.Linear(d_length, d_model)
        self.dropout = nn.Dropout(0.1)
        self.pe = nn.Parameter(torch.empty(1, number_shape, d_length))
        nn.init.uniform_(self.pe, -0.02, 0.02)
    def forward(self,input_embedding,fai_x, fai_x_prime,
                           w_1, b_1, w_2, b_2):
        query = self.layer_q(input_embedding)
        key = self.layer_k(input_embedding)
        value = self.layer_v(input_embedding)

        attn_weight = torch.bmm(query, torch.transpose(key, 1, 2))
        attn_weight = softmax(attn_weight, dim=-1)
        output = torch.bmm(attn_weight, value) + self.layer_add(input_embedding)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.layer_final(output)

        return output



class Transformer_learned_trig(nn.Module):
    def __init__(self, d_length,d_model,number_shape,number_class, m, device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')):
        super(Transformer_learned_trig, self).__init__()
        self.d_length = d_length
        self.d_model = d_model
        self.number_shape = number_shape
        self.number_class = number_class
        self.relu = nn.ReLU()

        self.layer_q = nn.Linear(d_length,d_model,bias=False)
        self.layer_k = nn.Linear(d_length, d_model, bias=False)
        self.layer_v = nn.Linear(d_length,d_model,bias=False)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_length, d_length)
        self.act = F.gelu
        self.layer_final = nn.Linear(d_model * number_shape, number_class)
        self.one = torch.unsqueeze(torch.eye(d_length),0).to(device)

        self.layer_add = nn.Linear(d_length, d_model)
        self.pe = nn.Parameter(torch.empty(1, number_shape, d_length))
        nn.init.uniform_(self.pe, -0.02, 0.02)
        self.dropout = nn.Dropout(0.1)
        ## addtional parameters for performer
        self.m = m
        self.device =device



    def forward(self,input_embedding,fai_x, fai_x_prime,
                           w_1, b_1, w_2, b_2,normalize=False):
        query = self.layer_q(input_embedding)
        key = self.layer_k(input_embedding)
        value = self.layer_v(input_embedding)



        batch_size, l, d = query.shape
        normalizer = 1 / (d ** 0.25) if normalize else 1

        m = self.m

        random_feats = orthogonal_gaussian(m, d)
        random_feats = torch.unsqueeze(random_feats, dim=0).expand(batch_size, m, d).to(self.device)

        def h(x):
            return torch.exp(torch.square(x).sum(dim=-1, keepdim=True) / 2)

        phi = phi_trig(h, m, random_feats)


        query_prime = phi(query * normalizer)
        key_prime = phi(key * normalizer)

        d_inv = torch.squeeze((torch.matmul(query_prime, torch.matmul(key_prime.transpose(1, 2), torch.ones((batch_size, l, 1)).to(self.device)))),
                              -1)
        d_inv = torch.diag_embed(1 / d_inv)

        output = torch.matmul(key_prime.transpose(1, 2), value)
        output = torch.matmul(query_prime, output)
        output = torch.matmul(d_inv, output)

        output = output + self.layer_add(input_embedding)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.layer_final(output)

        return output



class Transformer_learned_performer(nn.Module):
    def __init__(self, d_length,d_model,number_shape,number_class, m, device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')):
        super(Transformer_learned_performer, self).__init__()
        self.d_length = d_length
        self.d_model = d_model
        self.number_shape = number_shape
        self.number_class = number_class
        self.relu = nn.ReLU()

        self.layer_q = nn.Linear(d_length,d_model,bias=False)
        self.layer_k = nn.Linear(d_length, d_model, bias=False)
        self.layer_v = nn.Linear(d_length,d_model,bias=False)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_length, d_length)
        self.act = F.gelu
        self.layer_final = nn.Linear(d_model * number_shape, number_class)
        self.one = torch.unsqueeze(torch.eye(d_length),0).to(device)

        self.layer_add = nn.Linear(d_length, d_model)
        self.pe = nn.Parameter(torch.empty(1, number_shape, d_length))
        nn.init.uniform_(self.pe, -0.02, 0.02)
        self.dropout = nn.Dropout(0.1)
        ## addtional parameters for performer
        self.m = m
        self.device =device



    def forward(self,input_embedding,fai_x, fai_x_prime,
                           w_1, b_1, w_2, b_2,normalize=False):
        query = self.layer_q(input_embedding)
        key = self.layer_k(input_embedding)
        value = self.layer_v(input_embedding)



        batch_size, l, d = query.shape
        normalizer = 1 / (d ** 0.25) if normalize else 1

        m = self.m

        random_feats = orthogonal_gaussian(m, d)
        random_feats = torch.unsqueeze(random_feats, dim=0).expand(batch_size, m, d).to(self.device)

        def h(x):
            return torch.exp(-torch.square(x).sum(dim=-1, keepdim=True) / 2)

        phi = phi_postive(h, m, random_feats)


        query_prime = phi(query * normalizer)
        key_prime = phi(key * normalizer)

        d_inv = torch.squeeze((torch.matmul(query_prime, torch.matmul(key_prime.transpose(1, 2), torch.ones((batch_size, l, 1)).to(self.device)))),
                              -1)
        d_inv = torch.diag_embed(1 / d_inv)

        output = torch.matmul(key_prime.transpose(1, 2), value)
        output = torch.matmul(query_prime, output)
        output = torch.matmul(d_inv, output)

        output = output + self.layer_add(input_embedding)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.layer_final(output)

        return output

#Others class are for comparing the attention weights with transformer base




class ProbAttention(nn.Module):
    def __init__(self, d_length,d_model,number_shape,number_class, device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu'),
                 mask_flag=False, factor=20, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.d_length = d_length
        self.d_model = d_model
        self.number_shape = number_shape
        self.number_class = number_class
        self.relu = nn.ReLU()
        # self.layer_q = nn.Linear(d_length,d_model,bias=False)
        # self.layer_k = nn.Linear(d_length,d_model,bias=False)
        # w_q = torch.ones((d_length,d_model), requires_grad= True).to(device)
        # self.w_q = torch.nn.parameter(w_q)
        # w_k = torch.ones((d_length, d_length), requires_grad=True).to(device)
        # self.w_k = Parameter(w_k)
        self.layer_q = nn.Linear(d_length, d_model, bias=False)
        self.layer_k = nn.Linear(d_length, d_model, bias=False)
        self.layer_v = nn.Linear(d_length, d_model, bias=False)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_length, d_length)
        self.act = F.gelu
        self.layer_final = nn.Linear(d_model * number_shape, number_class)

        self.layer_add = nn.Linear(d_length, d_model)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape


        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self,input_embedding,fai_x, fai_x_prime,
                           w_1, b_1, w_2, b_2,normalize=False):

        B = input_embedding.shape[0]
        N = input_embedding.shape[1]
        h = 1
        queries = self.layer_q(input_embedding)          # batch * N * d
        keys = self.layer_k(input_embedding)            #
        values = self.layer_v(input_embedding)          #


        queries = queries.view(B, N, h, -1)
        keys = keys.view(B, N, h, -1)
        values = values.view(B, N, h, -1)

        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        attn_mask = None # assume

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        context = context.transpose(2, 1).contiguous()
        context = context.view(B, N, -1)

        context = context + self.layer_add(input_embedding)

        context = self.dropout(context)

        context = context.reshape(context.shape[0], -1)
        context = self.layer_final(context)

        return context

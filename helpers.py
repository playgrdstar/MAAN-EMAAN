import numpy as np
import pandas as pd
import scipy.sparse as sp

import json
from pathlib import Path
import networkx as nx
import torch as th
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, auc

import networkx as nx
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn


def load_multidata(edgelist_df, feature_df_1, feature_df_2, feature_df_3, feature_df_4, feature_df_5, threshold=1):

    assert edgelist_df.classes_encoded_cont.max()+1 == feature_df_1.shape[0]
    assert edgelist_df.classes_encoded_cont.max()+1 == feature_df_2.shape[0]
    assert edgelist_df.classes_encoded_cont.max()+1 == feature_df_3.shape[0]

    num_nodes = edgelist_df.classes_encoded_cont.max()+1
    screen_classes_edgelist = edgelist_df[['screen_encoded', 'classes_encoded_cont']]
    screen_classes_edgelist_grp = pd.DataFrame(screen_classes_edgelist.groupby(["screen_encoded", "classes_encoded_cont"]).size().reset_index(name='counts'))
    screen_classes_edgelist_grp.columns = ['screen_encoded', 'classes_encoded_cont', 'weight']
    filtered_edgelist = screen_classes_edgelist_grp[screen_classes_edgelist_grp.weight>threshold]
    print('Number of screen nodes: ', len(edgelist_df.screen_encoded.unique()))
    print('Number of class nodes: ', len(edgelist_df.classes_encoded_cont.unique()))

    adj_row = list(filtered_edgelist.screen_encoded) + list(filtered_edgelist.classes_encoded_cont)
    adj_col = list(filtered_edgelist.classes_encoded_cont) + list(filtered_edgelist.screen_encoded)
    num_edges = len(adj_row)
    print('Number of edges: ', num_edges)
    adj = sp.csc_matrix((np.ones(num_edges), (adj_row, adj_col)), shape=(num_nodes, num_nodes))
    # Features df -> sparse matrix
    feature_df_1 = feature_df_1.sort_values(by='id_encoded', axis=0)
    features_1 = sp.csr_matrix(feature_df_1.iloc[:,2:], dtype=np.float32)
    feature_df_2 = feature_df_2.sort_values(by='id_encoded', axis=0)
    features_2 = sp.csr_matrix(feature_df_2.iloc[:,2:], dtype=np.float32)
    feature_df_3 = feature_df_3.sort_values(by='id_encoded', axis=0)
    features_3 = sp.csr_matrix(feature_df_3.iloc[:,2:], dtype=np.float32)

    features_4 = sp.csr_matrix(feature_df_4.iloc[:,1:], dtype=np.float32)
    features_5 = sp.csr_matrix(feature_df_5.iloc[:,1:], dtype=np.float32)
    print('Dimensions of processed input features: ', features_1.shape, '|', features_2.shape, '|', features_3.shape,\
            '|', features_4.shape, '|', features_5.shape)
    return adj, features_1, features_2, features_3, features_4, features_5, num_nodes, num_edges

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0]) # add a self loop
    rowsum = np.array(adj_.sum(1)) # sum over rows
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten()) # sqrt of rowsum
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo() # A D^T D
    return adj_normalized

def mask_test_edges(adj, neg_ratio):
    adj_row = adj.nonzero()[0] # 1st row of indices of nodes in edges 
    adj_col = adj.nonzero()[1] # 2nd row of indices of nodes in edges
    edges = []
    edges_dic = {}
    for i in range(len(adj_row)):
        edges.append([adj_row[i], adj_col[i]]) # edgelist
        edges_dic[(adj_row[i], adj_col[i])] = 1 # dict of edgelist with 1
    false_edges_dic = {}
    num_test = int(np.floor(len(edges) / 10.))
    num_val = int(np.floor(len(edges) / 20.))
    # get and shuffle the edges
    all_edge_idx = np.arange(len(edges))
    np.random.shuffle(all_edge_idx)
    # get validation and test indices
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    edges = np.array(edges)
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    # delete extracted edges from the training dataset
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    # generate fake edges in validation and testing dataset
    test_edges_false = []
    val_edges_false = []
    while len(test_edges_false) < num_test*neg_ratio or len(val_edges_false) < num_val*neg_ratio:
        i = np.random.randint(0, adj.shape[0])
        # this needs to be changed to shape[1] if a rect matrix is involved
        j = np.random.randint(0, adj.shape[0])
        # skip if edges exist
        if (i, j) in edges_dic:
            continue
        if (j, i) in edges_dic:
            continue
        if (i, j) in false_edges_dic:
            continue
        if (j, i) in false_edges_dic:
            continue
        else:
            false_edges_dic[(i, j)] = 1
            false_edges_dic[(j, i)] = 1
        # probability of adding a false edge
        if np.random.random_sample() > 0.333 :
            if len(test_edges_false) < num_test*neg_ratio :
                test_edges_false.append((i, j))
            else:
                if len(val_edges_false) < num_val*neg_ratio :
                    val_edges_false.append([i, j])
        else:
            if len(val_edges_false) < num_val*neg_ratio :
                val_edges_false.append([i, j])
            else:
                if len(test_edges_false) < num_test*neg_ratio :
                    test_edges_false.append([i, j])
    
    # creating the training adj matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def mask_test_feas_continuous(feature, test_edges):
    test_nodes = list(set([node for edge in test_edges for node in edge]))
    train_feat = feature.copy()
    # train_feat[test_nodes] = np.random.randn(len(test_nodes), feature.shape[1])*1e-10
    train_feat[test_nodes] = np.zeros((len(test_nodes), feature.shape[1]))
    return train_feat

def mask_test_feas_binary(features, neg_ratio):
    fea_row = features.nonzero()[0]
    fea_col = features.nonzero()[1]
    feas = []
    feas_dic = {}
    for i in range(len(fea_row)):
        feas.append([fea_row[i], fea_col[i]])
        feas_dic[(fea_row[i], fea_col[i])] = 1
    false_feas_dic = {}
    num_test = int(np.floor(len(feas) / 10.))
    num_val = int(np.floor(len(feas) / 20.))
    all_fea_idx = np.arange(len(feas))
    np.random.shuffle(all_fea_idx)
    val_fea_idx = all_fea_idx[:num_val]
    test_fea_idx = all_fea_idx[num_val:(num_val + num_test)]
    feas = np.array(feas)
    test_feas = feas[test_fea_idx]
    val_feas = feas[val_fea_idx]
    train_feas = np.delete(feas, np.hstack([test_fea_idx, val_fea_idx]), axis=0)
    test_feas_false = []
    val_feas_false = []
    while len(test_feas_false) < num_test*neg_ratio or len(val_feas_false) < num_val*neg_ratio:
        i = np.random.randint(0, features.shape[0])
        j = np.random.randint(0, features.shape[1])
        if (i, j) in feas_dic:
            continue
        if (i, j) in false_feas_dic:
            continue
        else:
            false_feas_dic[(i, j)] = 1
        if np.random.random_sample() > 0.333 :
            if len(test_feas_false) < num_test*neg_ratio :
                test_feas_false.append([i, j])
            else:
                if len(val_feas_false) < num_val*neg_ratio :
                    val_feas_false.append([i, j])
        else:
            if len(val_feas_false) < num_val*neg_ratio :
                val_feas_false.append([i, j])
            else:
                if len(test_feas_false) < num_test*neg_ratio :
                    test_feas_false.append([i, j])
    data = np.ones(train_feas.shape[0])
    fea_train = sp.csr_matrix((data, (train_feas[:, 0], train_feas[:, 1])), shape=features.shape)
    return fea_train, train_feas, val_feas, val_feas_false, test_feas, test_feas_false

def compute_pos_norm(input_mat, type='adj'):
    if type == 'adj':
        input_mat_pos_weight = float(input_mat.shape[0] * input_mat.shape[0] - input_mat.sum()) / input_mat.sum()
        input_mat_norm = input_mat.shape[0] * input_mat.shape[0] / float((input_mat.shape[0] * input_mat.shape[0] - input_mat.sum()) * 2)
    elif type == 'feat':
        input_mat_ = sparse_to_tuple(input_mat.tocoo())
        input_mat_pos_weight = float(input_mat_[2][0] * input_mat_[2][1] - len(input_mat_[1])) / len(input_mat_[1])
        input_mat_norm = input_mat_[2][0] * input_mat_[2][1] / float((input_mat_[2][0] * input_mat_[2][1] - len(input_mat_[1])) * 2)

    return th.as_tensor(input_mat_pos_weight), th.as_tensor(input_mat_norm)

def log_likelihood_loss(labels, preds, criterion, norm=1):
    # labels = labels.flatten()
    cost = norm * criterion(input=preds, target=labels.flatten())
    return cost

def kl_loss(mean, log_std, num):
    kl = (0.5 / num) * th.mean(th.sum(1 + 2 * log_std - mean**2 - th.exp(log_std)**2, 1))
    return kl

def compute_rbf(x1,
                x2, latent_var,
                eps = 1e-7):
    """
    Computes the RBF Kernel between x1 and x2.
    """
    z_dim = x2.size(-1)
    sigma = 2. * z_dim * latent_var

    result = th.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
    return result

def compute_inv_mult_quad(x1,
                            x2, latent_var,
                            eps = 1e-7):
    """
    Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
    given by
            k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
    """
    z_dim = x2.size(-1)
    C = 2 * z_dim * latent_var
    kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))

    # Exclude diagonal elements
    result = kernel.sum() - kernel.diag().sum()

    return result

def compute_kernel(x1,
                    x2, latent_var,
                    kernel_type='rbf'):
    # Convert the tensors into row and column vectors
    D = x1.size(1)
    N = x1.size(0)

    x1 = x1.unsqueeze(-2) # Make it into a column tensor
    x2 = x2.unsqueeze(-3) # Make it into a row tensor

    """
    For when x1 and x2 have different sizes, expand along the 0th dimension.
    """
    x1 = x1.expand(N, N, D)
    x2 = x2.expand(N, N, D)

    if kernel_type == 'rbf':
        result = compute_rbf(x1, x2, latent_var)
    elif kernel_type == 'imq':
        result = compute_inv_mult_quad(x1, x2, latent_var)
    else:
        raise ValueError('Undefined kernel type.')

    return result

def compute_mmd(z, latent_var, size):
    # Sample from prior (Gaussian) distribution
    prior_z = th.randn_like(z)

    prior_z__kernel = compute_kernel(prior_z, prior_z, latent_var)
    z__kernel = compute_kernel(z, z, latent_var)
    priorz_z__kernel = compute_kernel(prior_z, z, latent_var)

    mmd = prior_z__kernel.mean() + \
            z__kernel.mean() - \
            2 * priorz_z__kernel.mean()

    bias_corr = size*(size-1)
    return mmd/bias_corr

def get_roc_score(num_nodes, edges_pos, edges_neg, reconstructions, adj_orig):

    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = reconstructions[0].reshape([num_nodes, num_nodes])
    adj_orig = adj_orig.todense()
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].detach().cpu().numpy()))
        pos.append(adj_orig[e[0], e[1]])
    
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].detach().cpu().numpy()))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.nan_to_num(np.hstack([preds, preds_neg]))
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    ps, rs, _ = precision_recall_curve(labels_all, preds_all)
    auc_score = auc(rs, ps)    

    return roc_score, ap_score, auc_score

def get_roc_score_a(num_nodes, feas_pos, feas_neg, reconstructions, features_orig):

    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1 + np.exp(-x))

    # Predict on test set of edges
    fea_rec = reconstructions.reshape([num_nodes, features_orig.shape[1]])
    features_orig = features_orig
    preds = []
    pos = []
    for e in feas_pos:
        preds.append(sigmoid(fea_rec[e[0], e[1]].detach().cpu().numpy()))
        pos.append(features_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in feas_neg:
        preds_neg.append(sigmoid(fea_rec[e[0], e[1]].detach().cpu().numpy()))
        neg.append(features_orig[e[0], e[1]])

    preds_all = np.nan_to_num(np.hstack([preds, preds_neg]))
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    ps, rs, _ = precision_recall_curve(labels_all, preds_all)
    auc_score = auc(rs, ps)   

    return roc_score, ap_score, auc_score

def get_rmse_recon(num_nodes, exc_edges, feat_recon, feat_orig_labels, num_features):
    feat_recon = feat_recon.reshape([num_nodes, num_features])
    feat_orig = feat_orig_labels.reshape([num_nodes, num_features])
    total_mse = 0
    total_nodes = len(exc_edges)*2
    for e in exc_edges:
        pred_0 = feat_recon[e[0]]
        pred_1 = feat_recon[e[1]]
        act_0 = feat_orig[e[0]]
        act_1 = feat_orig[e[1]]
        mse_0 = ((pred_0 - act_0) ** 2.).mean().item()
        mse_1 = ((pred_1 - act_1) ** 2.).mean().item()
        mse = mse_0 + mse_1
        total_mse += mse

    final_rmse = np.sqrt(total_mse/total_nodes)
    return final_rmse

def round_to_half(number):
    try:
        return round(number * 2) / 2
    except:
        return 0

def compute_weights(score):
    if score >= 4.75:
        return 16.0
    elif score < 4.75 and score >= 4.0:
        return 2.0
    elif score < 4.0 and score >= 3.5:
        return 2.0
    elif score <3.5 and score >= 3.0:
        return 4.0
    elif score < 3.0 and score >= 2.5:
        return 8.0
    elif score < 2.5 and score >= 2.0:
        return 16.0
    elif score < 2.0 and score >= 1.5:
        return 32.0
    elif score <= 1.5:
        return 128.0

def weighted_mse_loss(pred, target, weight):
    return th.mean(weight * (pred - target) ** 2)

def weighted_mae_loss(pred, target, weight):
    return th.mean(weight * th.abs(pred - target))
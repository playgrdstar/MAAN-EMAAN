
import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
import networkx as nx
import torch as th
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, auc
from tqdm import tqdm
import networkx as nx
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn

# Due to the stochasticity of the VAE framework, there will be some variability in the results
th.manual_seed(8)
np.random.seed(8)

device = th.device("cuda" if th.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings('ignore')

from models import RegressionModel
from helpers import load_multidata, sparse_to_tuple, \
                    preprocess_graph, mask_test_edges, mask_test_feas_continuous, \
                    mask_test_feas_binary, compute_pos_norm, log_likelihood_loss, \
                    kl_loss, compute_rbf, compute_inv_mult_quad, compute_kernel, compute_mmd, \
                    get_roc_score, get_roc_score_a, get_rmse_recon, round_to_half, compute_weights, \
                    weighted_mse_loss, weighted_mae_loss
                    


# load data
home_dir = Path(os.getcwd())
dataset = 'RICO_N' # this should correspond to the dataset, e.g., RICO_O
data_dir = home_dir/'data'/dataset
edgelist_filename = 'rico_edgelist.csv'
imagefeature_filename = 'imagefeature.csv'
wordfeature_filename = 'classnamefeature.csv'
description_filename = 'descfeature.csv'
edgelist_df = pd.read_csv(data_dir/edgelist_filename)
imagefeature_df = pd.read_csv(data_dir/imagefeature_filename)
wordfeature_df = pd.read_csv(data_dir/wordfeature_filename)
description_df = pd.read_csv(data_dir/description_filename)



# One hot encoding of genre components
total_num_nodes = edgelist_df['classes_encoded_cont'].max()+1
edgelist_df_screen_genre = edgelist_df[['screen_encoded', 'genre']]
edgelist_df_screen_genre = edgelist_df_screen_genre.drop_duplicates()
edgelist_df_screen_genre = edgelist_df_screen_genre.reset_index(drop=True)
genre_df = pd.get_dummies(edgelist_df_screen_genre['genre'], prefix=None)
genre_dim = genre_df.shape[1]
genre_df = pd.concat([edgelist_df_screen_genre, genre_df], axis=1)
genre_df = genre_df.drop(['genre'], axis=1)
genre_df = genre_df.sort_values(by='screen_encoded', axis=0)
genre_empty_df = pd.DataFrame(np.random.randn(total_num_nodes, genre_dim)*0, columns=genre_df.columns[1:])
genre_series = pd.DataFrame(range(total_num_nodes), columns=[genre_df.columns[0]])
genre_df_empty_df = pd.concat([genre_series, genre_empty_df], axis=1)
genre_df_empty_df[:len(genre_df)] = genre_df
expanded_genre_df = genre_df_empty_df.copy()


# One hot encoding of class components
edgelist_df_classes_comp = edgelist_df[['classes_encoded_cont', 'classes_comp']]
edgelist_df_classes_comp = edgelist_df_classes_comp.drop_duplicates()
edgelist_df_classes_comp = edgelist_df_classes_comp.reset_index(drop=True)
class_comp_df = pd.get_dummies(edgelist_df_classes_comp['classes_comp'], prefix=None)
class_comp_dim = class_comp_df.shape[1]
class_comp_df = pd.concat([edgelist_df_classes_comp, class_comp_df], axis=1)
class_comp_df = class_comp_df.drop(['classes_comp'], axis=1)
class_comp_df = class_comp_df.groupby('classes_encoded_cont').sum().reset_index(drop=False)
class_comp_df = class_comp_df.sort_values(by='classes_encoded_cont', axis=0)
class_comp_df = class_comp_df.set_index('classes_encoded_cont', drop=False)
class_comp_empty_df = pd.DataFrame(np.random.randn(total_num_nodes, class_comp_dim)*0, columns=class_comp_df.columns[1:])
class_comp_series = pd.DataFrame(range(total_num_nodes), columns=[class_comp_df.columns[0]])
class_comp_df_empty_df = pd.concat([class_comp_series, class_comp_empty_df], axis=1)
class_comp_df_empty_df[len(genre_df):] = class_comp_df
expanded_class_comp_df = class_comp_df_empty_df.copy()


edgelist_df_process = edgelist_df[['screen_encoded', 'classes_encoded_cont', 'score']]


adj, imagefeatures, wordfeatures, descfeatures, genrefeatures, compfeatures, num_nodes, num_edges = \
load_multidata(edgelist_df, imagefeature_df, wordfeature_df, 
                description_df, expanded_genre_df, expanded_class_comp_df, threshold=5)


adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, 10)


train_imagefeatures = mask_test_feas_continuous(imagefeatures, test_edges)
train_wordfeatures = mask_test_feas_continuous(wordfeatures, test_edges)
train_descfeatures = mask_test_feas_continuous(descfeatures, test_edges)

train_genrefeatures, train_genre, val_genre, val_genre_false, test_genre, test_genre_false = mask_test_feas_binary(genrefeatures, 10)
train_compfeatures, train_comp, val_comp, val_comp_false, test_comp, test_comp_false = mask_test_feas_binary(compfeatures, 10)


adj = adj_train
image_features_orig = imagefeatures
word_features_orig = wordfeatures
desc_features_orig = descfeatures
genre_features_orig = genrefeatures
comp_features_orig = compfeatures
adj_norm = preprocess_graph(adj)


pos_weight_u, norm_u = compute_pos_norm(adj, type='adj')
genre_pos_weight_a, genre_norm_a = compute_pos_norm(genrefeatures, type='feat')
comp_pos_weight_a, comp_norm_a = compute_pos_norm(compfeatures, type='feat')


adj = adj_norm
adj_label = adj_train + sp.eye(adj_train.shape[0]) # with self loops
adj_label = adj_label
image_features_label = image_features_orig
word_features_label = word_features_orig
desc_features_label = desc_features_orig
genre_features_label = genre_features_orig
comp_features_label = comp_features_orig


train_imagefeatures = th.FloatTensor(train_imagefeatures.todense())
train_wordfeatures = th.FloatTensor(train_wordfeatures.todense())
train_descfeatures = th.FloatTensor(train_descfeatures.todense())
train_genrefeatures = th.FloatTensor(train_genrefeatures.todense())
train_compfeatures = th.FloatTensor(train_compfeatures.todense())

adj_label = th.FloatTensor(adj_label.todense())

image_features_label = th.FloatTensor(image_features_label.todense())
word_features_label = th.FloatTensor(word_features_label.todense())
desc_features_label = th.FloatTensor(desc_features_label.todense())
genre_features_label = th.FloatTensor(genre_features_label.todense())
comp_features_label = th.FloatTensor(comp_features_label.todense())


nx_graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
adj = dgl.from_networkx(nx_graph) 


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_out,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.gcn1 = dglnn.GraphConv(in_feats, n_out, activation=activation)
        self.gcn2 = dglnn.GraphConv(n_out, n_out, activation=activation)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        h = self.gcn1(g,h)
        h = self.gcn2(g,h)
        return h

class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 n_out,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(dglnn.GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(dglnn.GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.fc = nn.Linear(in_features=num_hidden * heads[-2], 
                                    out_features=n_out, 
                                    bias=True)

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        out = self.fc(h)
        return out

class InnerDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, dropout=0., act=F.sigmoid):
        super(InnerDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, inputs):
        # z_u is the generated latent vector for the adjacency matrix
        # z_a is the generated latent vector for the feature matrix
        # z_u -> [num_nodes x hidden2_dim]
        # z_a -> [num_features x hidden2_dim]
        z_u, image_z_a, word_z_a, desc_z_a, genre_z_a, comp_z_a = inputs

        #adj
        z_u = self.dropout(z_u)
        z_u_t = z_u.T

        # [num_nodes x hidden2_dim] X [hidden2_dim X num_nodes]
        # x -> [num_nodes x num_nodes]
        adj_sq = th.matmul(z_u, z_u_t)

        #image
        # [num_nodes x hidden2_dim] X [hidden2_dim x num_features]
        # y -> [num_nodes x num_features]
        image_z_a = self.dropout(image_z_a)
        image_z_a_t = image_z_a.T
        image_rect = th.matmul(z_u, image_z_a_t)

        #word
        word_z_a = self.dropout(word_z_a)
        word_z_a_t = word_z_a.T
        word_rect = th.matmul(z_u, word_z_a_t)

        #desc
        desc_z_a = self.dropout(desc_z_a)
        desc_z_a_t = desc_z_a.T
        desc_rect = th.matmul(z_u, desc_z_a_t)

        #genre
        genre_z_a = self.dropout(genre_z_a)
        genre_z_a_t = genre_z_a.T
        genre_rect = th.matmul(z_u, genre_z_a_t)

        #comp
        comp_z_a = self.dropout(comp_z_a)
        comp_z_a_t = comp_z_a.T
        comp_rect = th.matmul(z_u, comp_z_a_t)

        adj_sq = self.act(adj_sq)
        image_rect = self.act(image_rect)
        word_rect = self.act(word_rect)
        desc_rect = self.act(desc_rect)
        genre_rect = self.act(genre_rect)
        comp_rect = self.act(comp_rect)

        # [1, num_nodes x num_nodes]
        edge_outputs = th.flatten(adj_sq)
        # [1, num_nodes x num_features]
        image_attri_outputs = th.flatten(image_rect)
        word_attri_outputs = th.flatten(word_rect)
        desc_attri_outputs = th.flatten(desc_rect)
        genre_attri_outputs = th.flatten(genre_rect)
        comp_attri_outputs = th.flatten(comp_rect)

        return edge_outputs, image_attri_outputs, word_attri_outputs, desc_attri_outputs, genre_attri_outputs, comp_attri_outputs

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = th.softmax(w, dim=1)

        return (beta * z).sum(1)

class FVAE(nn.Module):
    def __init__(self, dropout, 
                num_nodes,
                hidden1_dim, hidden2_dim):
        super(FVAE, self).__init__()

        self.num_nodes = num_nodes
        self.dropout = dropout
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        self.hidden_dense_a =  nn.Linear(in_features=self.num_nodes, 
                                    out_features=self.hidden1_dim, 
                                    bias=True)

        self.z_a_mean_dense = nn.Linear(in_features=self.hidden1_dim, 
                                    out_features=self.hidden2_dim, 
                                    bias=True)

        self.z_a_log_std_dense = nn.Linear(in_features=self.hidden1_dim, 
                                    out_features=self.hidden2_dim, 
                                    bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.1)

    def forward(self, features):
            
            # [num_features x num_nodes] X [num_nodes x hidden1_dim] -> [num_features x hidden1_dim]
            a = self.hidden_dense_a(features.T)
            a = F.tanh(a)

            # [num_features x hidden1_dim] X [hidden1_dim x hidden2_dim] -> [num_features x hidden2_dim]
            z_a_mean = self.z_a_mean_dense(a)
            z_a_log_std = self.z_a_log_std_dense(a)

            # [num_features x hidden2_dim]
            z_a = z_a_mean + th.randn(features.shape[1], self.hidden2_dim).to(device) * th.exp(z_a_log_std)

            return z_a, z_a_mean, z_a_log_std

class MGVAE(nn.Module):
    def __init__(self, dropout, 
                image_num_features, word_num_features, desc_num_features, genre_num_features, comp_num_features,
                hidden1_dim, hidden2_dim):
        super(MGVAE, self).__init__()

        self.image_feature_dim = image_num_features
        self.word_feature_dim = word_num_features
        self.desc_feature_dim = desc_num_features
        self.genre_feature_dim = genre_num_features
        self.comp_feature_dim = comp_num_features

        self.dropout = dropout
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        # for adj - forward will take in the feature inputs
        # [num_nodes x num_features] x [num_features x num_hidden] -> [num_nodes x num_hidden]
        self.image_hidden_conv_u = GAT(num_layers=4,
                                    in_dim = self.image_feature_dim,
                                    num_hidden = self.hidden1_dim,
                                    n_out=self.hidden2_dim,
                                    heads=[1,1,1,1],
                                    activation=F.relu,
                                    feat_drop=self.dropout,
                                    attn_drop=self.dropout,
                                    negative_slope=0.1,
                                    residual=False)

        self.word_hidden_conv_u = GAT(num_layers=4,
                                    in_dim = self.word_feature_dim,
                                    num_hidden = self.hidden1_dim,
                                    n_out=self.hidden2_dim,
                                    heads=[1,1,1,1],
                                    activation=F.relu,
                                    feat_drop=self.dropout,
                                    attn_drop=self.dropout,
                                    negative_slope=0.1,
                                    residual=False)

        self.desc_hidden_conv_u = GAT(num_layers=4,
                                    in_dim = self.desc_feature_dim ,
                                    num_hidden = self.hidden1_dim,
                                    n_out=self.hidden2_dim,
                                    heads=[1,1,1,1],
                                    activation=F.relu,
                                    feat_drop=self.dropout,
                                    attn_drop=self.dropout,
                                    negative_slope=0.1,
                                    residual=False)

        self.genre_hidden_conv_u = GAT(num_layers=4,
                                    in_dim = self.genre_feature_dim ,
                                    num_hidden = self.hidden1_dim,
                                    n_out=self.hidden2_dim,
                                    heads=[1,1,1,1],
                                    activation=F.relu,
                                    feat_drop=self.dropout,
                                    attn_drop=self.dropout,
                                    negative_slope=0.1,
                                    residual=False)
    
        self.comp_hidden_conv_u = GAT(num_layers=4,
                                in_dim = self.comp_feature_dim ,
                                num_hidden = self.hidden1_dim,
                                n_out=self.hidden2_dim,
                                heads=[1,1,1,1],
                                activation=F.relu,
                                feat_drop=self.dropout,
                                attn_drop=self.dropout,
                                negative_slope=0.1,
                                residual=False)


        self.z_u_mean_conv = GCN(in_feats=self.hidden2_dim, 
                        n_out=self.hidden2_dim, 
                        n_layers=1, activation=lambda x: x, dropout=self.dropout)

        self.z_u_log_std_conv = GCN(in_feats=self.hidden2_dim, 
                n_out=self.hidden2_dim, 
                n_layers=1, activation=lambda x: x, dropout=self.dropout)


        self.semantic_attention = SemanticAttention(in_size=self.hidden2_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.001)


    def forward(self, adj, imagefeatures, wordfeatures, descfeatures, genrefeatures, compfeatures):

            # [num_nodes x num_features] X [num_features x hidden1_dim] -> [num_nodes x hidden1_dim]
            image_u = self.image_hidden_conv_u(adj, imagefeatures)
            word_u = self.word_hidden_conv_u(adj, wordfeatures)
            desc_u = self.desc_hidden_conv_u(adj, descfeatures)
            genre_u = self.genre_hidden_conv_u(adj, genrefeatures)
            comp_u = self.comp_hidden_conv_u(adj, compfeatures)

            semantic_embeddings = []
            semantic_embeddings.append(image_u)
            semantic_embeddings.append(word_u)
            semantic_embeddings.append(desc_u)
            semantic_embeddings.append(genre_u)
            semantic_embeddings.append(comp_u)

            semantic_embeddings = th.stack(semantic_embeddings, dim=1) 
            u = self.semantic_attention(semantic_embeddings)  

            # [num_nodes x hidden1_dim] X [hidden1_dim x hidden2_dim] -> [num_nodes x hidden2_dim]
            z_u_mean = self.z_u_mean_conv(adj, u)
            z_u_log_std = self.z_u_log_std_conv(adj, u)

            # [num_nodes x hidden2_dim]
            z_u = z_u_mean + th.randn(imagefeatures.shape[0], self.hidden2_dim).to(device) * th.exp(z_u_log_std)

            return u, semantic_embeddings, z_u, z_u_mean, z_u_log_std

class MAAN_NET(nn.Module):
    def __init__(self, dropout, 
                image_num_features, word_num_features, desc_num_features, genre_num_features, comp_num_features, num_nodes,
                hidden1_dim, hidden2_dim):
        super(MAAN_NET, self).__init__()

        self.image_feature_dim = image_num_features
        self.word_feature_dim = word_num_features
        self.desc_feature_dim = desc_num_features
        self.genre_feature_dim = genre_num_features
        self.comp_feature_dim = comp_num_features

        self.num_nodes = num_nodes
        self.dropout = dropout
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        self.mgvae = MGVAE(self.dropout, self.image_feature_dim, self.word_feature_dim, self.desc_feature_dim, 
                            self.genre_feature_dim, self.comp_feature_dim,
                            self.hidden1_dim, self.hidden2_dim)

        self.image_fvae = FVAE(self.dropout, self.num_nodes, self.hidden1_dim, self.hidden2_dim)
        self.word_fvae = FVAE(self.dropout, self.num_nodes, self.hidden1_dim, self.hidden2_dim)
        self.desc_fvae = FVAE(self.dropout, self.num_nodes, self.hidden1_dim, self.hidden2_dim)
        self.genre_fvae = FVAE(self.dropout, self.num_nodes, self.hidden1_dim, self.hidden2_dim)
        self.comp_fvae = FVAE(self.dropout, self.num_nodes, self.hidden1_dim, self.hidden2_dim)

        self.decoder = InnerDecoder(act = lambda x: x)
        
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.001)


    def forward(self, adj, imagefeatures, wordfeatures, descfeatures, genrefeatures, compfeatures):

            u, semantic_embeddings, z_u, z_u_mean, z_u_log_std = self.mgvae(adj, imagefeatures, wordfeatures, descfeatures, 
                                                                            genrefeatures, compfeatures)

            image_z_a, image_z_a_mean, image_z_a_log_std = self.image_fvae(imagefeatures)
            word_z_a, word_z_a_mean, word_z_a_log_std = self.word_fvae(wordfeatures)
            desc_z_a, desc_z_a_mean, desc_z_a_log_std = self.desc_fvae(descfeatures)
            genre_z_a, genre_z_a_mean, genre_z_a_log_std = self.genre_fvae(genrefeatures)
            comp_z_a, comp_z_a_mean, comp_z_a_log_std = self.comp_fvae(compfeatures)

            reconstructions = self.decoder((z_u, image_z_a, word_z_a, desc_z_a, genre_z_a, comp_z_a))

            # train_reg_out = self.reg(z_u[train_screen_id])
            # test_reg_out = self.reg(z_u[test_screen_id])

            return u, semantic_embeddings, \
                        z_u, z_u_mean, z_u_log_std, \
                        image_z_a, image_z_a_mean, image_z_a_log_std, \
                        word_z_a, word_z_a_mean, word_z_a_log_std, \
                        desc_z_a, desc_z_a_mean, desc_z_a_log_std, \
                        genre_z_a, genre_z_a_mean, genre_z_a_log_std, \
                        comp_z_a, comp_z_a_mean, comp_z_a_log_std, \
                        reconstructions




learning_rate = 0.02
n_epochs = 1000 # for training the main model
reg_n_epochs = 5000 # for separate regression training
hidden1_dim = 64
hidden2_dim = 32
weight_decay = 0.001
dropout = 0.1
kernel_type = 'rbf'
latent_var = 32

image_num_features = imagefeatures.shape[1]
word_num_features = wordfeatures.shape[1]
desc_num_features = descfeatures.shape[1]
genre_num_features = genrefeatures.shape[1]
comp_num_features = compfeatures.shape[1]

model = MAAN_NET(dropout, image_num_features, word_num_features, desc_num_features, 
                genre_num_features, comp_num_features, num_nodes, hidden1_dim, hidden2_dim)


model = model.to(device)
train_imagefeatures = train_imagefeatures.to(device)
train_wordfeatures = train_wordfeatures.to(device)
train_descfeatures = train_descfeatures.to(device)
train_genrefeatures = train_genrefeatures.to(device)
train_compfeatures = train_compfeatures.to(device)
adj_label= adj_label.to(device)
image_features_label = image_features_label.to(device)
word_features_label = word_features_label.to(device)
desc_features_label = desc_features_label.to(device)
genre_features_label = genre_features_label.to(device)
comp_features_label = comp_features_label.to(device)
norm_u = norm_u.to(device)
genre_norm_a = genre_norm_a.to(device)
comp_norm_a = comp_norm_a.to(device)
adj = adj.to(device)

params = list(model.parameters())
optimizer = th.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs//2)
criterion_u = th.nn.BCEWithLogitsLoss(pos_weight=pos_weight_u, reduction='mean')
image_criterion_a = nn.MSELoss()
word_criterion_a = nn.MSELoss()
desc_criterion_a = nn.MSELoss()
genre_criterion_a = th.nn.BCEWithLogitsLoss(pos_weight=genre_pos_weight_a, reduction='mean')
comp_criterion_a = th.nn.BCEWithLogitsLoss(pos_weight=comp_pos_weight_a, reduction='mean')
score_criterion = nn.MSELoss()

best_valid_roc = best_valid_ap = best_valid_auc = 0
best_test_roc = best_test_ap = best_test_auc = 0
best_epoch = 0

print('Training started.')
for epoch in tqdm(range(n_epochs)):
    scheduler.step()
    model.train()
    optimizer.zero_grad()

    u, semantic_embeddings, z_u, z_u_mean, z_u_log_std, \
    image_z_a, image_z_a_mean, image_z_a_log_std, \
    word_z_a, word_z_a_mean, word_z_a_log_std, \
    desc_z_a, desc_z_a_mean, desc_z_a_log_std, \
    genre_z_a, genre_z_a_mean, genre_z_a_log_std, \
    comp_z_a, comp_z_a_mean, comp_z_a_log_std, \
    reconstructions = model(adj, train_imagefeatures, train_wordfeatures, train_descfeatures, 
                                                        train_genrefeatures, train_compfeatures)

    preds_u = reconstructions[0]
    image_preds_a = reconstructions[1]
    word_preds_a = reconstructions[2]
    desc_preds_a = reconstructions[3]
    genre_preds_a = reconstructions[4]
    comp_preds_a = reconstructions[5]

    loss_kl_u = kl_loss(z_u_mean, z_u_log_std, num_nodes)
    loss_kl = loss_kl_u 

    loss_recon_adj = log_likelihood_loss(adj_label, preds_u, criterion_u, norm_u)
    loss_recon_image = log_likelihood_loss(image_features_label, image_preds_a, image_criterion_a)
    loss_recon_word = log_likelihood_loss(word_features_label, word_preds_a, word_criterion_a)
    loss_recon_desc = log_likelihood_loss(desc_features_label, desc_preds_a, desc_criterion_a)
    loss_recon_genre = log_likelihood_loss(genre_features_label, genre_preds_a, genre_criterion_a, genre_norm_a)
    loss_recon_comp = log_likelihood_loss(comp_features_label, comp_preds_a, comp_criterion_a, comp_norm_a)
    loss_recon = loss_recon_adj + loss_recon_image + loss_recon_word + loss_recon_desc + loss_recon_genre + loss_recon_comp

    loss_mmd_image = compute_mmd(image_z_a, latent_var, image_num_features)
    loss_mmd_word = compute_mmd(word_z_a, latent_var, word_num_features)
    loss_mmd_desc = compute_mmd(desc_z_a, latent_var, desc_num_features)
    loss_mmd_genre = compute_mmd(genre_z_a, latent_var, genre_num_features)
    loss_mmd_comp = compute_mmd(comp_z_a, latent_var, comp_num_features)
    loss_mmd = loss_mmd_image + loss_mmd_word + loss_mmd_desc + loss_mmd_genre + loss_mmd_comp

    loss = loss_kl + loss_recon + loss_mmd

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        roc_curr, ap_curr, auc_curr = get_roc_score(num_nodes, val_edges, val_edges_false, reconstructions, adj_orig)
        roc_score, ap_score, auc_score = get_roc_score(num_nodes, test_edges, test_edges_false, reconstructions, adj_orig)

        if roc_curr > best_valid_roc and ap_curr > best_valid_ap: 
            best_valid_roc = roc_curr
            best_valid_ap = ap_curr
            best_valid_auc = auc_curr

            if roc_score > best_test_roc and ap_score > best_test_ap and auc_score > best_test_auc:

                best_test_roc = roc_score
                best_test_ap = ap_score
                best_test_auc = auc_score

                best_z_u = z_u.cpu().detach().numpy().copy()

print('Link Prediction - ROC-AUC: ', best_test_roc, '| AP Edges: ', best_test_ap)






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

from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

from dgl.base import DGLError
# from dgl.transform import reverse

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

class GATConvConcat(nn.Module):
    def __init__(self,
                 in_feats,
                 edge_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super().__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        
        # added for edge weights
        self.edgefc = nn.Linear(edge_feats, out_feats * num_heads, bias=False)

        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))

        # added for edge weight
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)`')

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1] # num_nodes
                # print(f'src_prefix_shape shape: {src_prefix_shape}')
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                # print(f'feat_src shape: {feat_src.shape}')

                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            
            # added for edge weight
            # [num_edges, num_heads, hidden_dim]
            # print('Edge weights shape: ', graph.edata['weight'].shape)
            edge_ft = self.edgefc(graph.edata['weight'].float().unsqueeze(-1)).view(-1, self._num_heads, self._out_feats)
            # print('edge_ft shape: ', edge_ft.shape)
            
            # feat_src: [num_nodes, num_heads, hidden_dim]
            # attn_l: [1, num_heads, hidden_dim]
            # feat_src * attn_l -> [num_nodes, num_heads, hidden_dim]
            # sum -> [num_nodes, num_heads]
            # unsqueeze -> [num_nodes, num_heads, 1]
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            # e -> [num_edges, num_heads, 1]

            e = graph.edata.pop('e')

            # added for edge weight
            e = self.leaky_relu(e + edge_ft)
            # print('e shape: ', e.shape)
            # e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            # a -> [num_edges, num_heads, 1]
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            # feat_src: [.., num_heads, hidden_dim] x [.., num_heads, 1]
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            # ft: [num_nodes, num_heads, hidden_dim]
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
            # Still [num_nodes, num_heads, hidden_dim], which is flattened to num_heads*hidden_dim later

class GATConvMul(nn.Module):
    def __init__(self,
                 in_feats,
                 edge_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super().__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        
        # added for edge weights
        self.edgefc = nn.Linear(edge_feats, out_feats * num_heads, bias=False)

        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))

        # added for edge weight
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)`')

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1] # num_nodes
                # print(f'src_prefix_shape shape: {src_prefix_shape}')
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                # print(f'feat_src shape: {feat_src.shape}')

                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            
            # added for edge weight
            # [num_edges, num_heads, hidden_dim]
            # print('Edge weights shape: ', graph.edata['weight'].shape)
            edge_ft = self.edgefc(graph.edata['weight'].float().unsqueeze(-1)).view(-1, self._num_heads, self._out_feats)
            # print('edge_ft shape: ', edge_ft.shape)
            
            # feat_src: [num_nodes, num_heads, hidden_dim]
            # attn_l: [1, num_heads, hidden_dim]
            # feat_src * attn_l -> [num_nodes, num_heads, hidden_dim]
            # sum -> [num_nodes, num_heads]
            # unsqueeze -> [num_nodes, num_heads, 1]
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            # e -> [num_edges, num_heads, 1]

            e = graph.edata.pop('e')

            # added for edge weight -> change this to multiply?
            e = self.leaky_relu(e * edge_ft)
            # print('e shape: ', e.shape)
            # e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            # a -> [num_edges, num_heads, 1]
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            # feat_src: [.., num_heads, hidden_dim] x [.., num_heads, 1]
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            # ft: [num_nodes, num_heads, hidden_dim]
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
            # Still [num_nodes, num_heads, hidden_dim], which is flattened to num_heads*hidden_dim later


class RegressionModel(nn.Module):
    def __init__(self, dim):
        super(RegressionModel, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(self.dim, self.dim//2, bias=True)
        self.fc2 = nn.Linear(self.dim//2, self.dim//4, bias=True)
        self.fc3 = nn.Linear(self.dim//4, self.dim//8, bias=True)
        self.fc4 = nn.Linear(self.dim//8, self.dim//16, bias=True)
        self.fc5 = nn.Linear(self.dim//16, 1, bias=False)

        self.dropout = nn.Dropout(0.01, inplace=True)

    def forward(self, embeddings):
        out = self.fc1(embeddings)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = self.dropout(out)
        out = self.fc5(out)
        
        return out
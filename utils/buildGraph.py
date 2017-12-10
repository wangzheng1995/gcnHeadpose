import numpy as np
from scipy.io import loadmat
import os
import math
from collections import defaultdict
import networkx as nx
import scipy.sparse as sp

cent_dist = 6

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def construct_feed_dict(features, support, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    #return adj
    inv_sum = sp.diags((1.0/rowsum).flatten())
    return adj.dot(inv_sum).transpose().tocoo()
    #d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    #d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    #d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    #return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_feat(feat):
    """Preprocessing of features"""
    feat = sp.coo_matrix(feat)
    return sparse_to_tuple(feat)

def _build_graph(frames):
    idx = 0
    num_frame = len(frames)
    #print ("num_frame", num_frame)
    d = defaultdict(list)
    for i in range(num_frame): # range of frame
        num_box = frames[i]['box_num'][0][0]
        idx_head = idx
        d[idx] = []
        for j in range(num_box): # range of box, connection within frame
            idx = idx_head + j
            for j1 in range(j+1, num_box):
                d[idx].append(idx_head+j1)
            if i < num_frame-1: # connection between frame
                best_dist = pow(100,2)
                s = -1
                #print (i,j)
                #print(frames[i]['box'][0])
                
                sz = frames[i]['box'][0][j]['size'][0]/2.0
                bx = np.array((frames[i]['box'][0][j]['x'][0]+sz,
                        frames[i]['box'][0][j]['y'][0]+sz))
                for k in range(frames[i+1]['box_num'][0]):
                    sz2 = frames[i+1]['box'][0][k]['size'][0]/2.0
                    bx2 = np.array((frames[i+1]['box'][0][k]['x'][0]+sz2,
                            frames[i+1]['box'][0][k]['y'][0]+sz2))
                    dist = np.linalg.norm(bx-bx2)
                    if dist < best_dist:
                        best_dist = dist
                        s = k
                if s > -1 and best_dist < cent_dist:
                    d[idx].append(idx_head+num_box+s)
                label = frames[i]['box'][0][j]['label'][0]
                p = frames[i]['box'][0][j]['p'][0]
            idx = idx_head + num_box
    return d

def get_feature_label(frames):
    num_frame = len(frames)
    features = []
    labels = []
    for i in range(num_frame): 
        num_box = frames[i]['box_num'][0][0]
        for j in range(num_box):
            p = frames[i]['box'][0][j]['p'][0]
            label = frames[i]['box'][0][j]['label'][0][0]
            features.append(p)
            labels.append(label)
    return np.asarray(features), np.asarray(labels)

def prepare():
    path = "/DATACENTER/9/wangzheng/highfive/svm_prob"
    set1_m1 = os.path.join(path,"set1_m1_data.mat")
    m = loadmat(set1_m1)
    step = 5
    #print len(m['frm'][0]) # 150
    #print len(m['frm'][0][0][2][0]) # 34
    #print len(m['frm'][0][0][2][0][0][1][0]) # 4
    sample_feat = []
    sample_lbl = []
    sample_adj = []
    for i in range(len(m['frm'][0])):
        #print i
        j = 0
        for j in range(int(math.ceil(float(m['frm'][0][i]['frame_num'][0])/5.0))): # frame_num
            if (j+1)*step > int(m['frm'][0][i]['frame_num'][0]):
                graph_sample = m['frm'][0][i]['frame'][0][j*step:int(m['frm'][0][i]['frame_num'][0])]
            else:
                graph_sample = m['frm'][0][i][2][0][j*step:(j+1)*step]
            graph_dict = _build_graph(graph_sample)
            adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict))
            features, labels = get_feature_label(graph_sample)
            #print features.shape
            #print labels.shape
            #print adj
            sample_feat.append(features)
            sample_lbl.append(labels)
            sample_adj.append(adj)
    return sample_feat, sample_lbl, sample_adj
    
def load_raw_data():
    feat_file = "/DATACENTER/9/wangzheng/highfive/hog_conv_64/set1_m1/features.txt"
    label_file = "/DATACENTER/9/wangzheng/highfive/hog_conv_64/set1_m1/labels.txt"
    f1 = open(label_file, 'r')
    f2 = open(feat_file, 'r')

    x = []
    lbls = []
    for l in f1.readlines():
        lbl = l.split()
        
        break

#    for l in f2.readlines():
#        arr = np.array((map(float, l.split())))       
#        break
            

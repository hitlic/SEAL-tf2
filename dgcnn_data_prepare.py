# coding: utf-8
"""
@author Liuchen
2019
"""

import networkx as nx
import numpy as np
import random
import tensorflow as tf


class Graph(nx.Graph):
    """
    辅助类，用于定义一个样本图数据，并提供处理工具
    """

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)
        self.label = None

    @property
    def features(self):
        """
        网络中所有节点特征构成的矩阵
        """
        node_list = sorted(list(self.nodes))
        feature_dict = nx.get_node_attributes(self, 'feature')
        if not feature_dict:
            return None

        g_feautres = [feature_dict[node] for node in node_list]
        return np.stack(g_feautres)

    @property
    def feature_dim(self):
        nodes = self.node_list
        if self.nodes[nodes[0]].get('feature', None) is None:
            return 0
        return len(self.nodes[nodes[0]]['feature'])

    @property
    def degree_list(self):
        degree_dict = dict(self.degree)
        return [degree_dict[k] for k in sorted(degree_dict.keys())]

    @property
    def edge_list(self):
        return list(self.edges)

    @property
    def node_list(self):
        return sorted(self.nodes)

    @property
    def num_nodes(self):
        return self.number_of_nodes()


def create_graph(g_txt):
    """
    根据一段文本，创建一个网络。节点可以有特征，也可以无特征
    Args:
        g_text:文本片段，格式如下（节点有特征）
                网络标签
                邻节点1 邻节点2 : 特征1 特征2 特征3
                邻节点1 邻节点2 : 特征1 特征2 特征3
                邻节点1 邻节点2 : 特征1 特征2 特征3
            或者（节点无特征）：
                网络标签
                邻节点1 邻节点2
                邻节点1 邻节点2
                邻节点1 邻节点2
    Return: 网络
    """
    g = Graph()
    g_lines = g_txt.split('\n')
    g_label = int(g_lines[0].strip())
    node_lines = [l for l in g_lines[1:] if l.strip() != '']
    for i, nf in enumerate(node_lines):
        ns_feat = nf.split(':')
        if len(ns_feat) > 1:
            ns, feat = ns_feat
            feat = [float(n) for n in feat.split()]
            g.add_node(i, feature=feat)
        else:
            ns = ns_feat[0]
            g.add_node(i)
        
        for n in ns.split():
            g.add_edge(i, int(n))
    g.label = g_label
    return g


def onehot(n, dim):
    one_hot = [0] * dim
    one_hot[n] = 1
    return one_hot


def load_data(f_path, test_percent=.0):
    """
    读取文件，返回训练集、测试集

    文件中，每个网络的首行为 “G 网络标签”，以后每行中“:”前为当前行表示的节点的邻节点，“:”后为节点特征
        -----------格式开始----------
        G 网络1标签
        邻节点1 邻节点2 ... : 特征1 特征2 特征3 ...
        邻节点1 邻节点2 ... : 特征1 特征2 特征3 ...
        邻节点1 邻节点2 ... : 特征1 特征2 特征3 ...
        G 网络2标签
        邻节点1 邻节点2 ... : 特征1 特征2 特征3 ...
        邻节点1 邻节点2 ... : 特征1 特征2 特征3 ...
        邻节点1 邻节点2 ... : 特征1 特征2 特征3 ...
        ... ...
        -----------格式结束----------
    """
    f = open(f_path)
    graphs_txt = f.read()
    f.close()
    graph_strs = graphs_txt.split('G')
    graphs = []
    for g_str in graph_strs:
        if g_str == '':
            continue
        g = create_graph(g_str)
        graphs.append(g)

    labels = {g.label for g in graphs}
    class_num = len(labels)
    label_dict = {l: i for i, l in enumerate(labels)}

    # 将图标签转为 one-hot 形式
    for g in graphs:
        g.label = onehot(label_dict[g.label], class_num)

    random.shuffle(graphs)
    if test_percent == .0:
        return graphs
    else:
        test_num = int(len(graphs) * test_percent)
        test = graphs[0:test_num]
        train = graphs[test_num:]
        return train, test


def batching(graph_batch):
    """
    处理一个batch的图
    :graph_batch: 图list
    :return: (batch_features, batch_ajacent, batch_degree_inv, graph_indexes, batch_label)
        batch_features 当前batch中所有图中节点特征
        batch_ajacent 稀疏矩阵，当前batch中所有图的邻接矩阵组成的大分块邻接矩阵
        batch_degree_inv 稀疏矩阵，ajacent对应的度矩阵的逆矩阵
        graph_indexes 当前batch中每个图的特征在features 中的起始和结束位置
        batch_label 当前batch中图的标签
    """
    # 若无节点特征，则以节点的度为特征
    if graph_batch[0].features is not None:
        batch_features = [g.features for g in graph_batch]
        batch_features = np.concatenate(batch_features, 0).astype(np.float32)
    else:
        batch_features = [g.degree_list for g in graph_batch]
        batch_features = np.concatenate(batch_features, 0).reshape([-1, 1]).astype(np.float32)

    # 每个图的特征的索引开始位置和结束位置
    g_node_nums = [g.num_nodes for g in graph_batch]
    graph_indexes = [[sum(g_node_nums[0:i-1]), sum(g_node_nums[0:i])] for i in range(1, len(g_node_nums)+1)]
    graph_indexes = np.array(graph_indexes, dtype=np.int32)

    # 图标签
    batch_label = [g.label for g in graph_batch]
    batch_label = np.array(batch_label)

    # batch中所有网络节点的degree列表
    batch_degrees = [d for g in graph_batch for d in g.degree_list]
    total_node_num = len(batch_degrees)

    # 邻接矩阵的稀疏矩阵
    indices = []
    for i, g in enumerate(graph_batch):
        start_pos = graph_indexes[i][0]
        g_nodes = g.node_list
        for e in g.edges:
            node_from = start_pos + g_nodes.index(e[0])
            node_to = start_pos + g_nodes.index(e[1])
            indices.append([node_from, node_to])
            indices.append([node_to, node_from])
    values = np.ones(len(indices), dtype=np.float32)
    indices = np.array(indices, dtype=np.int32)
    shape = np.array([total_node_num, total_node_num], dtype=np.int32)
    batch_ajacent = tf.sparse.SparseTensor(indices, values, shape)

    # 度矩阵的逆的稀疏矩阵
    index_degree_inv = [([i, i], 1.0/degree if degree > 0 else 0) for i, degree in enumerate(batch_degrees)]
    index_degree_inv = list(zip(*index_degree_inv))
    batch_degree_inv = tf.sparse.SparseTensor(index_degree_inv[0], index_degree_inv[1], shape)

    return batch_features, batch_ajacent, batch_degree_inv, graph_indexes, batch_label


if __name__ == '__main__':
    gs = load_data('./data/MUTAG.txt')
    a, d_inv, f, l, ind = batching(gs)
    print(len(gs), ind.shape, a.shape, f.shape)
    print(ind)

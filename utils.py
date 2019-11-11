# coding: utf-8
"""
@author Liuchen
2019

部分代码参考 https://github.com/muhanzhang/SEAL
"""

import networkx as nx
import random as rand
import node2vec as nv
import numpy as np
import time
from tqdm import tqdm
from scipy.sparse import csr_matrix
import scipy.sparse as ssp
import multiprocessing as mp
from dgcnn_data_prepare import Graph, onehot
import copy


def subgraph_extraction_labeling(ind, A, h, max_nodes_per_hop=None):
    """
    抽取一个链接的封闭子图
    extract the h-hop enclosing subgraph around link 'ind'
    """
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None and max_nodes_per_hop > 0:
            if max_nodes_per_hop < len(fringe):
                fringe = rand.sample(fringe, max_nodes_per_hop)
        if not fringe:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    subgraph = A[nodes, :][:, nodes]
    # apply node-labeling
    labels = node_label(subgraph)
    return nodes, labels.tolist(), ind


def neighbors(fringe, A):
    """
    find all 1-hop neighbors of nodes in fringe from A
    """
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


def node_label(subgraph):
    """
    对一个封闭子图进行 Double-Radius Node Labeling
    """
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels > 1e6] = 0  # set inf labels to 0
    labels[labels < -1e6] = 0  # set -inf labels to 0
    return labels


def parallel_worker(x):
    return subgraph_extraction_labeling(*x)


def extract_subgraph_from_links(g, links, h, max_nodes_per_hop=None, multi_process=False, info=''):
    """
    抽取列表中所有链接的封闭子图
    Args:
        g: networkX Graph
        links: 节点对
        h: hop number
        max_nodes_per_hop: 每个hop中最大节点数
        multi_process: 是否使用多进程（单进程，速度慢，可调试; 多进程，速度快，不可调试，Windows下不可用）
        info: 进度条显示信息
    """
    # 根据网络构造稀疏矩阵
    row, col = zip(*g.edges)
    data = np.ones(2*g.number_of_edges(), dtype=np.int)
    node_num = g.number_of_nodes()
    A = csr_matrix((data, (row+col, col+row)), shape=(node_num, node_num))

    # 单进程，速度慢，可调试
    if not multi_process:
        g_list = []
        for i, j in tqdm(links, desc=f"{info}-subgraph abstracting"):
            nodes, labels, link = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop)
            gl_dict = {nodes[n]: labels[n] for n in range(len(nodes))}
            sub_g = copy.deepcopy(Graph(g.subgraph(nodes)))
            nx.set_node_attributes(sub_g, gl_dict, 'dr_label')
            if sub_g.has_edge(link[0], link[1]):
                sub_g.remove_edge(link[0], link[1])
            g_list.append(sub_g)
        return g_list
    else:
        # 多进程，速度快，不可调试，Windows下不可用
        pool = mp.Pool(mp.cpu_count())
        results = pool.map_async(parallel_worker, [((i, j), A, h, max_nodes_per_hop) for i, j in links])
        remaining = results._number_left  # pylint: disable=protected-access
        pbar = tqdm(total=remaining, desc=f"{info}-subgraph abstracting")
        while True:
            pbar.update(remaining - results._number_left)  # pylint: disable=protected-access
            if results.ready():
                break
            remaining = results._number_left  # pylint: disable=protected-access
            time.sleep(1)
        results = results.get()
        pool.close()
        pbar.close()
        g_list = []
        for nodes, labels, link in results:
            gl_dict = {nodes[n]: labels[n] for n in range(len(nodes))}
            sub_g = copy.deepcopy(Graph(g.subgraph(nodes)))
            nx.set_node_attributes(sub_g, gl_dict, 'dr_label')
            if sub_g.has_edge(link[0], link[1]):
                sub_g.remove_edge(link[0], link[1])
            g_list.append(sub_g)
        return g_list


def enclosed_subgraph(g, hop=1, max_hop_nodes=None, link_percent=1.,
                      embedding_dim=0, has_feature=False, inject_neg_links=True, multi_process=False):
    """
    抽取封闭子图
    Args:
        g: networkx Graph
        hop: 最大hop
        max_hop_nodes: 每个hop中，最大节点数量
        link_percent: g 中用于预测的边的比例
        has_feature: 是否使用节点特征，如果为True则每个节点有属性"has_feature"
        embedding: 是否使用node2vec生成每个节点的embedding
        multi_process: 是否使用多进程（单进程，速度慢，可调试; 多进程，速度快，不可调试，Windows下不可用）
    Return: 正例子图列表, 负例子图列表
    """
    # 正例链接
    pos_links = list(g.edges)
    pos_links = rand.sample(pos_links, int(g.number_of_edges() * link_percent))
    num_pos_links = len(pos_links)

    # 负例链接
    neg_links = []
    nodes = list(g.nodes)
    i = 0
    while True:
        node1 = rand.choice(nodes)
        node2 = rand.choice(nodes)
        if node1 == node2 or g.has_edge(node1, node2):
            continue
        neg_links.append((node1, node2))
        i += 1
        if i >= num_pos_links:
            break

    # 加入节点node2vec embedding
    if embedding_dim > 0:
        print("node2vec embedding ... ...")
        if inject_neg_links:  # 是否加入neg_links
            g.add_edges_from(neg_links)
        n2v_model = nv.Node2Vec(g, dimensions=embedding_dim, walk_length=30, num_walks=10, workers=4)
        n2v_wv = n2v_model.fit().wv
        nv_dict = {int(n): v for n, v in zip(n2v_wv.index2word, n2v_wv.vectors)}
        if not has_feature:
            nx.set_node_attributes(g, nv_dict, 'feature')
        else:
            feat_dict = nx.get_node_attributes(G, 'feature')
            features = {n: np.concatenate([feat_dict[n], nv_dict[n]]) for n in feat_dict.keys()}
            nx.set_node_attributes(g, features, 'feature')
        if inject_neg_links:
            g.remove_edges_from(neg_links)

    # 抽取封闭子图
    pos_sub_gs = extract_subgraph_from_links(g, pos_links, hop, max_hop_nodes, multi_process, info='pos')
    neg_sub_gs = extract_subgraph_from_links(g, neg_links, hop, max_hop_nodes, multi_process, info='neg')

    # 处理子图的 Double-Radius Node Label
    dr_label_set = set()
    for sg in pos_sub_gs:
        sg.label = [1, 0]
        dr_label_set = dr_label_set.union(set(nx.get_node_attributes(sg, 'dr_label').values()))
    for sg in neg_sub_gs:
        sg.label = [0, 1]
        dr_label_set = dr_label_set.union(set(nx.get_node_attributes(sg, 'dr_label').values()))
    dr_label_dict = {v: i for i, v in enumerate(list(dr_label_set))}
    dr_label_dim = len(dr_label_set)

    # 在节点特征中加入 Double-Radius Node
    for gs in pos_sub_gs:
        for n in gs.nodes:
            dr_l = gs.nodes[n]['dr_label']
            if has_feature or embedding_dim > 0:
                feat = gs.nodes[n]['feature']
                gs.nodes[n]['feature'] = np.concatenate(
                    [feat, onehot(dr_label_dict[dr_l], dr_label_dim)]).astype(np.float32)
            else:
                gs.nodes[n]['feature'] = np.array(onehot(dr_label_dict[dr_l], dr_label_dim)).astype(np.float32)
    for gs in neg_sub_gs:
        for n in gs.nodes:
            dr_l = gs.nodes[n]['dr_label']
            if has_feature or embedding_dim > 0:
                feat = gs.nodes[n]['feature']
                gs.nodes[n]['feature'] = np.concatenate(
                    [feat, onehot(dr_label_dict[dr_l], dr_label_dim)]).astype(np.float32)
            else:
                gs.nodes[n]['feature'] = np.array(onehot(dr_label_dict[dr_l], dr_label_dim)).astype(np.float32)
    return pos_sub_gs, neg_sub_gs


def create_dataset(g, val_num=0, test_num=0, hop=1, max_hop_nodes=None, link_percent=1., has_feature=False,
                   embedding_dim=0, inject_neg_links=True, multi_process=False):
    """
    抽取封闭子图，构建训练、验证、测试集
    Args:
        g: networX Graph
        val_num: 验证集数量
        test_num: 测试集数量
        hop: hop number
        max_hop_nodes: 每个hop中最大节点数量
        link_percent: 使用的网络链接数量
        has_feature: 网络g中的节点是否包含了特征 'feature'
        embedding_dim: node2vec 维度，0表示不使用embedding
        inject_neg_links: embedding时是否使用负样本链接（不存在的链接）
        multi_process: 是否使用多进程（单进程，速度慢，可调试; 多进程，速度快，不可调试，Windows下不可用）
    Return: (训练集，验证集，测试集)
    """
    print("抽取封闭子图 ... ... ... ... ... ...")
    pos_data, neg_data = enclosed_subgraph(g, hop, max_hop_nodes, link_percent, embedding_dim,
                                           has_feature, inject_neg_links, multi_process)
    data_set = pos_data + neg_data
    rand.shuffle(data_set)

    test = data_set[:test_num]
    val = data_set[test_num: test_num + val_num]
    train = data_set[test_num + val_num:]
    return train, val, test


if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([
        (0, 2), (0, 3), (0, 4), (1, 4), (1, 5), (1, 6),
        (2, 3), (4, 5),
        (2, 7), (2, 8), (3, 8), (3, 9), (3, 10), (4, 10), (4, 11), (5, 11), (5, 12), (5, 13), (6, 13), (6, 14),
        (7, 8), (9, 10), (11, 12), (13, 14),
        (7, 15), (7, 16), (8, 16), (8, 17), (9, 17), (9, 18), (10, 18), (10, 19),
        (11, 19), (11, 20), (12, 20), (12, 21), (13, 21), (13, 22), (14, 22)
    ])
    # G = nx.generators.erdos_renyi_graph(100, 0.1)
    nx.set_node_attributes(G, {i: [i, i] for i in G.nodes}, 'has_feature')

    # sub_g = link_enclosed_subgraph(G, [0, 1], 2)
    # print(sub_g.nodes[22])

    gsp, gsn = enclosed_subgraph(G, hop=2, max_hop_nodes=0, link_percent=1.0, embedding_dim=0, has_feature=True)

    print([g.num_nodes for g in gsp])
    print([g.num_nodes for g in gsn])

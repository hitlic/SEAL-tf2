# coding: utf-8
"""
@author Liuchen
2019
"""

import networkx as nx
import utils
import dgcnn_models as dgcnn

# G = nx.generators.erdos_renyi_graph(300, 0.1)  # nx 生成ER随机网络
G = nx.windmill_graph(100, 5)  # nx生成的一种特殊结构的网络

# 使用说明：
#         把数据构造出一个networkx 的Graphs，替换这里的G即可。若每个节点有属性向量，可作为各节点的 “feature” 属性，
#     然后在create_dataset方法中将has_feature设为True即可。
#             例： G.nodes[0]['feature'] = [0., 1., 0., 1.]
#                 为G中的节点0添加了"feature"属性，值为[0., 1., 0., 1.]
#     注意：要么所有节点都没有feature，要么全都有且长度一致

train_set, val_set, test_set = utils.create_dataset(G, val_num=100, test_num=100, hop=1, max_hop_nodes=None,
                                                    link_percent=1.0, has_feature=False,
                                                    embedding_dim=64, inject_neg_links=True, multi_process=True)
class_num = len(train_set[0].label)        # 类别数量
feature_dim = train_set[0].feature_dim     # 节点特征维度
k = 19                                     # 每个网络输出的最重要的top_k个节点
model = dgcnn.create_model(class_num, feature_dim, k)
# model.summary()

dgcnn.train(model, train_set, val_set, epochs=5, learing_rate=0.01, batch_size=32)
print(dgcnn.test(test_set, model))
print(dgcnn.predict(test_set, model))

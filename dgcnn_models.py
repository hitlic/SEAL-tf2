# coding: utf-8
"""
@author Liuchen
2019
"""

import tensorflow as tf
from tensorflow import keras
import dgcnn_data_prepare as dp
import math
import tqdm


class GCNLayer(keras.layers.Layer):
    def __init__(self, out_dim, *args, **kwargs):
        """
        Args:
            out_dim: 输出特征维度
        """
        self.out_dim = out_dim
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        in_dim = input_shape[0][1]

        self.W = self.add_weight(shape=(in_dim, self.out_dim),
                                 initializer='uniform', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        """
        原文式(2)
        Args:
            inputs: [input_Z, batch_ajacent, batch_degree_inv]
                input_Z: 特征
                batch_ajacent: 稀疏矩阵，一个batch的graph构成的大矩阵
                batch_degree_inv: 稀疏矩阵，一个batch的grapha的大度矩阵的逆矩阵
        """
        input_Z, batch_ajacent, batch_degree_inv = inputs
        AZ = tf.sparse.sparse_dense_matmul(batch_ajacent, input_Z)   # AZ
        AZ = tf.add(AZ, input_Z)                                    # AZ+Z = (A+I)Z
        AZW = tf.matmul(AZ, self.W)                                 # (A+I)ZW
        DAZW = tf.sparse.sparse_dense_matmul(batch_degree_inv, AZW)  # D^-1AZW
        return tf.nn.tanh(DAZW)  # tanh 激活

    def compute_output_shape(self, input_shape):
        node_num = input_shape[0][0]
        return (node_num, self.out_dim)


class SortPooling(keras.layers.Layer):
    def __init__(self, k, *args, **kwargs):
        """
        Args:
            k: 每个网络的输出大小，原文中2.2节SortPooling层中的 k
            graph_indexes:  一个batch中各个graph在大矩阵中的位置
        """
        super().__init__(*args, **kwargs)
        self.k = k

    def build(self, input_shape):
        super().build(input_shape)
        self.total_dim = input_shape[0][1]

    def call(self, inputs):
        """
        Args:
            inputs: [features, graph_indexes]
        """
        features, graph_indexes = inputs

        def sort_a_graph(index_span):
            indices = tf.range(index_span[0], index_span[1])  # 获取单个图的节点特征索引
            graph_feature = tf.gather(features, indices)        # 获取单个图的全部节点特征

            graph_size = index_span[1] - index_span[0]
            k = tf.cond(self.k > graph_size, lambda: graph_size, lambda: self.k)  # k与图size比较
            # 根据最后一列排序，返回前k个节点的特征作为图的表征
            top_k = tf.gather(graph_feature, tf.nn.top_k(graph_feature[:, -1], k=k).indices)

            # 若图size小于k，则补0行
            zeros = tf.zeros([self.k - k, self.total_dim], dtype=tf.float32)
            top_k = tf.concat([top_k, zeros], 0)
            return top_k

        sort_pooling = tf.map_fn(sort_a_graph, graph_indexes, dtype=tf.float32)
        return sort_pooling

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[1][0]
        return (batch_size, self.k, self.total_dim)


def create_model(class_num, feature_dim, k,
                 gcnn_dims=(32, 32, 32, 1),  # 各DGCNN层特征维度
                 conv1d_1_filters=16,        # 第一个一维卷积层卷积核数量
                 conv1d_2_filters=32,        # 第二个一维卷积层卷积核数量
                 conv1d_2_kernel=5,          # 第二个一维卷积层卷积核大小
                 dense_out_dim=128,          # 全连接层维度
                 drop_out=0.5):              # drop_out 比例

    # 全连接层输入维度，基于前几层参数计算得到
    dense_in_dim = (int((k - 2) / 2 + 1) - conv1d_2_kernel + 1) * conv1d_2_filters

    # 四个输入，分别是一个batch的特征、邻接矩阵、度矩阵逆、图位置索引
    inputs_f = keras.layers.Input(batch_shape=[None, feature_dim])
    inputs_a = keras.layers.Input(batch_shape=[None, None], sparse=True)
    inputs_d = keras.layers.Input(batch_shape=[None, None], sparse=True)
    inputs_i = keras.layers.Input(batch_shape=[None, 2], dtype=tf.int32)
    inputs = [inputs_f, inputs_a, inputs_d, inputs_i]

    # 四个DGCNN层
    outputs1 = GCNLayer(out_dim=gcnn_dims[0])(inputs[:3])
    outputs2 = GCNLayer(out_dim=gcnn_dims[1])([outputs1, inputs[1], inputs[2]])
    outputs3 = GCNLayer(out_dim=gcnn_dims[2])([outputs2, inputs[1], inputs[2]])
    outputs4 = GCNLayer(out_dim=gcnn_dims[3])([outputs3, inputs[1], inputs[2]])
    outputs = keras.layers.Concatenate(-1)([outputs1, outputs2, outputs3, outputs4])
    # SortPooling层
    outputs = SortPooling(k=k)([outputs, inputs[3]])
    outputs = keras.layers.Reshape([-1, 1])(outputs)
    # 两个一维卷积层
    outputs = keras.layers.Conv1D(filters=conv1d_1_filters, kernel_size=sum(gcnn_dims), strides=sum(gcnn_dims))(outputs)
    outputs = keras.layers.ReLU()(outputs)
    outputs = keras.layers.MaxPool1D(pool_size=2, strides=2)(outputs)
    outputs = keras.layers.Conv1D(filters=conv1d_2_filters, kernel_size=conv1d_2_kernel, strides=1)(outputs)
    outputs = keras.layers.ReLU()(outputs)
    outputs = keras.layers.Reshape([dense_in_dim])(outputs)
    # 全连接层
    outputs = keras.layers.Dense(dense_out_dim, activation='relu')(outputs)
    outputs = keras.layers.Dropout(drop_out)(outputs)
    # 输出层
    outputs = keras.layers.Dense(class_num, activation="softmax")(outputs)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def run_batch(graphs, model, opt=None, acc=None, train=True):
    """
    训练或验证一个batch
    """
    features, ajacent, dgree_inv, graph_index, labels = dp.batching(graphs)

    with tf.GradientTape() as tape:
        predict = model([features, ajacent, dgree_inv, graph_index])
        loss = tf.reduce_mean(keras.losses.categorical_crossentropy(labels, predict))
    if train:
        grad = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients(zip(grad, model.trainable_weights))
        acc.update_state(tf.argmax(labels, 1), tf.argmax(predict, 1))
        return loss.numpy(), acc.result().numpy()
    else:
        acc_val = tf.keras.metrics.Accuracy()
        acc_val.update_state(tf.argmax(labels, 1), tf.argmax(predict, 1))
        return loss.numpy(), acc_val.result().numpy(), tf.argmax(predict, 1).numpy()


def train_a_batch(graphs, model, opt, acc):
    return run_batch(graphs, model, opt, acc)


def validate(graphs, model):
    loss, acc, _ = run_batch(graphs, model, train=False)
    return loss, acc

def test(graphs, model):
    loss, acc, _ = run_batch(graphs, model, train=False)
    return loss, acc

def predict(graphs, model):
    _, _, pred = run_batch(graphs, model, train=False)
    return pred


def train(model, train_graphs, validate_graphs=None, epochs=10, learing_rate=0.05, batch_size=32):
    opt = tf.optimizers.Adam(learning_rate=learing_rate)
    acc = keras.metrics.Accuracy()

    # 数据分batch
    batchs = []
    batch_num = math.ceil(len(train_graphs) / batch_size)
    for i in range(batch_num):
        batch = train_graphs[i*batch_size: (i+1)*batch_size]
        batchs.append(batch)

    # 迭代epochs
    for e in range(epochs):
        acc.reset_states()
        # 迭代batchs
        tqdm_batchs = tqdm.tqdm(batchs, bar_format="{bar}{r_bar}|{percentage:3.0f}% {desc}")
        for batch_gs in tqdm_batchs:
            train_loss, train_acc = train_a_batch(batch_gs, model, opt, acc)
            # print(e, train_loss, train_acc)
            tqdm_batchs.set_description(f'epoch: {e + 1} - loss: {train_loss:.6}, acc: {train_acc:.6}')
        if validate_graphs:
            val_loss, val_acc = validate(validate_graphs, model)
            print(f'epoch-{e + 1:3}  loss: {train_loss:.6}, acc: {train_acc:.6}, val_loss: {val_loss:.6}, val_acc: {val_acc:.6}')
    return model, opt

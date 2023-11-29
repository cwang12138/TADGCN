"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2023/9/9 22:10
# @Author  : Chen Wang
# @Site    : 
# @File    : tadgcn.py
# @Email   : chen.wang@ahnu.edu.cn
# @details : Time-Aware Attention-Based Dynamic Graph Convolution Network (TADGCN)
*****************************************************
"""
import numpy as np
import tensorflow as tf
from model.tjmte_cell import TJMTECell
from model.adagcn_cell import ADAGCNCell
from einops import repeat


class TADGCN(tf.keras.Model):
    def __init__(self, static_node_embeddings, args):
        super().__init__()
        self.args = args
        if args.is_multi_view:
            self.closeness_length = args.closeness_length
            self.trend_length = args.trend_length
            self.period_length = args.period_length
            self.seq_in = args.closeness_length + args.trend_length + args.period_length
            self.seq_out = args.prediction_length
        else:
            self.closeness_length = args.closeness_length
            self.seq_in = args.closeness_length
            self.seq_out = args.prediction_length
        self.static_node_embeddings = static_node_embeddings
        self.stack_num = args.stack_num
        self.cheb_k = args.cheb_k
        self.tjmte_heads = args.tjmte_heads
        self.tjmte_head_dim = args.tjmte_head_dim
        self.dropout = args.dropout
        self.num_node = args.sensor_size
        self.agcn_out_dim = args.agcn_out_dim
        self.batch_size = args.batch_size
        self.tjmte_dim = args.tjmte_dim
        self.dynamic_time_embed_dim = args.dynamic_time_embed_dim
        self.dynamic_node_embed_dim = args.dynamic_node_embed_dim
        self.static_node_embed_dim = args.static_node_embed_dim
        self.gfe_dim = args.gfe_dim
        self.dmaf_heads = args.dmaf_heads
        self.dmaf_head_dim = args.dmaf_head_dim
        self.closeness_use_dynamic = args.closeness_use_dynamic
        self.trend_use_dynamic = args.trend_use_dynamic
        self.period_use_dynamic = args.period_use_dynamic
        self.num_graph = 0

        # init dynamic adaptive node embedding
        if self.closeness_use_dynamic:
            self.closeness_node_embeddings = tf.Variable(
                tf.random.normal([self.closeness_length, self.num_node, self.dynamic_node_embed_dim])
            )       # (P, N, D)
            self.num_graph = self.num_graph + self.closeness_length
            # init closeness dynamic adaptive graph feature embedding
            self.graph_feature_embed_closeness = tf.Variable(tf.random.normal([self.closeness_length, 1, self.gfe_dim]))
        else:
            self.closeness_node_embeddings = tf.Variable(
                tf.random.normal([self.num_node, self.dynamic_node_embed_dim])
            )       # (N, D)
            self.num_graph = self.num_graph + 1
            # init closeness dynamic adaptive graph feature embedding
            self.graph_feature_embed_closeness = tf.Variable(tf.random.normal([1, 1, self.gfe_dim]))

        if self.trend_use_dynamic:
            self.trend_node_embeddings = tf.Variable(
                tf.random.normal([self.trend_length, self.num_node, self.dynamic_node_embed_dim])
            )       # (Q, N, D)
            self.num_graph = self.num_graph + self.trend_length
            # init trend dynamic adaptive graph feature embedding
            self.graph_feature_embed_trend = tf.Variable(tf.random.normal([self.trend_length, 1, self.gfe_dim]))
        else:
            self.trend_node_embeddings = tf.Variable(
                tf.random.normal([self.num_node, self.dynamic_node_embed_dim])
            )       # (N, D)
            self.num_graph = self.num_graph + 1
            # init trend dynamic adaptive graph feature embedding
            self.graph_feature_embed_trend = tf.Variable(tf.random.normal([1, 1, self.gfe_dim]))

        if self.period_use_dynamic:
            self.period_node_embeddings = tf.Variable(
                tf.random.normal([self.period_length, self.num_node, self.dynamic_node_embed_dim])
            )       # (Q, N, D)
            self.num_graph = self.num_graph + self.period_length
            # init period dynamic adaptive graph feature embedding
            self.graph_feature_embed_period = tf.Variable(tf.random.normal([self.period_length, 1, self.gfe_dim]))
        else:
            self.period_node_embeddings = tf.Variable(
                tf.random.normal([self.num_node, self.dynamic_node_embed_dim])
            )       # (N, D)
            self.num_graph = self.num_graph + 1
            # init period dynamic adaptive graph feature embedding
            self.graph_feature_embed_period = tf.Variable(tf.random.normal([1, 1, self.gfe_dim]))

        # init model components
        self.TADGCN = []
        for i in range(self.stack_num):
            self.TADGCN.append(ADAGCNCell(closeness_len=self.closeness_length,
                                          target_len=self.trend_length,
                                          cheb_k=self.cheb_k,
                                          agcn_input_dim=1,
                                          agcn_output_dim=self.agcn_out_dim,
                                          embed_dim=self.dynamic_node_embed_dim + self.static_node_embed_dim,
                                          heads=self.dmaf_heads,
                                          head_dim=self.dmaf_head_dim,
                                          dropout=self.dropout,
                                          num_node=self.num_node,
                                          closeness_use_dynamic=self.closeness_use_dynamic,
                                          trend_use_dynamic=self.trend_use_dynamic,
                                          period_use_dynamic=self.period_use_dynamic))

            self.TADGCN.append(TJMTECell(dimension=self.tjmte_dim,
                                         seq_in=self.seq_in,
                                         seq_out=self.seq_out,
                                         heads=self.tjmte_heads,
                                         head_dim=self.tjmte_head_dim,
                                         dropout=self.dropout,
                                         is_multi_view=args.is_multi_view))

            if i + 1 != self.stack_num:
                self.TADGCN.append(tf.keras.layers.Conv2D(self.seq_in, (1, self.tjmte_dim), use_bias=True))

        self.time_projection = tf.keras.layers.Dense(self.dynamic_time_embed_dim)
        self.output_layer = tf.keras.layers.Conv2D(self.seq_out, (1, self.tjmte_dim), use_bias=True)

    def call(self, inputs, *args, **kwargs):
        # inputs: x_flow, (B, N, P); x_time, (B, P, 5); y_time, (B, Q, 5)
        x_flow, x_time, y_time = inputs

        # dynamic node embedding concat static node embedding
        if self.closeness_use_dynamic:
            closeness_node_embeddings = tf.concat(
                [self.closeness_node_embeddings,
                 repeat(self.static_node_embeddings[tf.newaxis, ...], '() n d -> p n d', p=self.closeness_length)],
                axis=-1
            )   # (P, N, dynamic_node_embed_dim + static_node_embed_dim)
            graph_feature_embed_closeness = self.graph_feature_embed_closeness
        else:
            closeness_node_embeddings = tf.concat([self.closeness_node_embeddings, self.static_node_embeddings], axis=-1)
            graph_feature_embed_closeness = repeat(self.graph_feature_embed_closeness, '() n d -> p n d',
                                                   p=self.closeness_length)

        if self.trend_use_dynamic:
            trend_node_embeddings = tf.concat(
                [self.trend_node_embeddings,
                 repeat(self.static_node_embeddings[tf.newaxis, ...], '() n d -> p n d', p=self.trend_length)],
                axis=-1
            )   # (Q, N, dynamic_node_embed_dim + static_node_embed_dim)
            graph_feature_embed_trend = self.graph_feature_embed_trend
        else:
            trend_node_embeddings = tf.concat([self.trend_node_embeddings, self.static_node_embeddings], axis=-1)
            graph_feature_embed_trend = repeat(self.graph_feature_embed_trend, '() n d -> p n d',
                                               p=self.trend_length)

        if self.period_use_dynamic:
            period_node_embeddings = tf.concat(
                [self.period_node_embeddings,
                 repeat(self.static_node_embeddings[tf.newaxis, ...], '() n d -> p n d', p=self.period_length)],
                axis=-1
            )   # (Q, N, dynamic_node_embed_dim + static_node_embed_dim)
            graph_feature_embed_period = self.graph_feature_embed_period
        else:
            period_node_embeddings = tf.concat([self.period_node_embeddings, self.static_node_embeddings], axis=-1)
            graph_feature_embed_period = repeat(self.graph_feature_embed_period, '() n d -> p n d',
                                                p=self.period_length)

        graph_feature_embeddings = tf.concat(
            [graph_feature_embed_period, graph_feature_embed_trend, graph_feature_embed_closeness], axis=0
        )       # (P + 2Q, N, D)

        x_time_embed = self.time_projection(x_time)  # (B, P, dynamic_time_embed_dim)

        # output: (B, N, P, agcn_output_dim)
        current_outputs = []
        for idx in range(self.stack_num):
            input_flow = x_flow
            x_flow = self.TADGCN[idx * 3]([x_flow, closeness_node_embeddings, trend_node_embeddings,
                                           period_node_embeddings, graph_feature_embeddings])

            x_flow = tf.concat(
                [x_flow, repeat(x_time_embed[:, tf.newaxis], 'b () p d -> b n p d', n=self.num_node)], axis=-1
            )
            x_flow, current_output = self.TADGCN[idx * 3 + 1]([x_flow, x_time, y_time])
            current_outputs.append(current_output)

            if idx != (self.stack_num - 1):
                x_flow = tf.squeeze(self.TADGCN[idx * 3 + 2](tf.transpose(x_flow, (0, 1, 3, 2))))    # (B, N, P)
                x_flow = x_flow + input_flow  # residual

        final_output = tf.squeeze(self.output_layer(tf.transpose(x_flow, (0, 1, 3, 2))))
        current_outputs.append(final_output)
        current_outputs = tf.stack(current_outputs, axis=0)                       # (stack_num, B, N, Q)
        predict = tf.reduce_sum(current_outputs, axis=0)                          # (B, N, Q)
        return predict


if __name__ == '__main__':
    pass

"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2023/9/9 22:11
# @Author  : Chen Wang
# @Site    : 
# @File    : adagcn_cell.py
# @Email   : chen.wang@ahnu.edu.cn
# @details : Attention-Driven Dynamic Adaptive Graph Convolution Network (ADAGCN) Cell
*****************************************************
"""
import tensorflow as tf
from einops import rearrange, repeat


# Adaptive Graph Convolution class
class AGCN(tf.keras.Model):
    def __init__(self, cheb_k, input_dim, output_dim, embed_dim, num_node):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = tf.Variable(tf.random.normal([embed_dim, cheb_k, input_dim, output_dim]))
        self.bias_pool = tf.Variable(tf.random.normal([embed_dim, output_dim]))
        self.num_node = num_node

    def call(self, input, *args, **kwargs):
        # input contains [x_flow, node_embeds]
        # x_flow shape: (B, N, input_dim), node_embeds shape: (N, D)
        # output shape: (B, N, output_dim)
        x, node_embedding = input
        supports = tf.nn.softmax(tf.nn.relu(tf.einsum('i d, j d -> i j', node_embedding, node_embedding)), axis=1)  # T1
        identity_matrix = tf.eye(self.num_node)      # T0
        support_set = [identity_matrix, supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(tf.matmul(2 * supports, support_set[-1]) - support_set[-2])  # Tk = 2xTk-1 - Tk-2, T1=x
        supports = tf.stack(support_set, axis=0)    # (cheb_k, N, N)
        weights = tf.einsum('n d, d k i o -> n k i o', node_embedding, self.weights_pool)
        bias = tf.einsum('n d, d o -> n o', node_embedding, self.bias_pool)
        x_g = tf.einsum('k n m, b m c -> b k n c', supports, x)     # (B, cheb_k, N, input_dim)
        x_g = tf.transpose(x_g, (0, 2, 1, 3))       # (B, N, cheb_k, input_dim)
        x_gconv = tf.einsum('b n k i, n k i o -> b n o', x_g, weights) + bias
        return x_gconv    # (B, N, output_dim)


# Pre-LayerNormalization class
class PreNorm(tf.keras.Model):
    def __init__(self, fn):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization()
        self.fn = fn

    def call(self, inputs, *args, **kwargs):
        if not isinstance(inputs, list):
            return self.fn(self.norm(inputs), **kwargs)
        else:
            inputs_norm = [self.norm(x) for x in inputs]
            return self.fn(inputs_norm, **kwargs)


# Self Attention class
class Attention(tf.keras.Model):
    def __init__(self, dimension, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dimension)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = tf.keras.layers.Dense(inner_dim * 3, use_bias=False)
        self.to_out = tf.keras.Sequential([
            tf.keras.layers.Dense(dimension),
            tf.keras.layers.Dropout(dropout)
        ]) if project_out else tf.identity

    def call(self, x, *args, **kwargs):
        # x.shape = (B, P, D)
        b, p, d, h = *x.shape, self.heads
        qkv = tf.split(self.to_qkv(x), axis=-1, num_or_size_splits=3)
        q, k, v = map(lambda t: rearrange(t, 'b p (h d) -> b h p d', h=h), qkv)

        # q, k, v shape: (B H P D)
        dots = tf.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attention = tf.nn.softmax(dots, axis=-2)

        out = tf.einsum('b h i j, b h j d -> b h i d', attention, v)
        out = rearrange(out, 'b h p d -> b p (h d)')
        out = self.to_out(out)
        return out      # (B, P, D)


# Cross Attention class
class CrossAttention(tf.keras.Model):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = tf.keras.layers.Dense(inner_dim, use_bias=False)
        self.to_v = tf.keras.layers.Dense(inner_dim, use_bias=False)
        self.to_q = tf.keras.layers.Dense(inner_dim, use_bias=False)

        self.to_out = tf.keras.Sequential([
            tf.keras.layers.Dense(dim),
            tf.keras.layers.Dropout(dropout)
        ]) if project_out else tf.identity

    def call(self, inputs, **kwargs):
        # inputs: [q, kv], output shape is consistent with q
        # node-to-graph: q - graph feature token; k, v - node feature
        # graph-to-node: q - node feature; k, v: graph feature
        # graph feature embedding: (1, P, 1, D); node feature: (B, P, N, D)
        q, kv = inputs

        k = self.to_k(kv)
        k = rearrange(k, 'b p n (h d) -> b h p n d', h=self.heads)

        v = self.to_v(kv)
        v = rearrange(v, 'b p n (h d) -> b h p n d', h=self.heads)

        q = self.to_q(q)
        q = rearrange(q, 'b p n (h d) -> b h p n d', h=self.heads)

        dots = tf.einsum('b h p i d, b h p j d -> b h p i j', q, k) * self.scale

        attention = tf.nn.softmax(dots, axis=-1)

        out = tf.einsum('b h p i j, b h p j d -> b h p i d', attention, v)
        out = rearrange(out, 'b h p n d -> b p n (h d)')
        out = self.to_out(out)      # g2n: (B, P, N, D); n2g: (B, P, 1, D)
        return out


# Dynamic Multi-Graph Attention Fusion class
class DMAF(tf.keras.Model):
    def __init__(self, dimension, heads, head_dim, dropout):
        super().__init__()
        self.node_2_graph = PreNorm(CrossAttention(dimension, heads, head_dim, dropout))
        self.multi_graph_attention_fusion = PreNorm(Attention(dimension, heads, head_dim, dropout))
        self.graph_2_node = PreNorm(CrossAttention(dimension, heads, head_dim, dropout))
        self.gft_projection = tf.keras.layers.Dense(dimension)

    def call(self, inputs, *args, **kwargs):
        # inputs: (B, P, N, agcn_output_dim); (P, 1, gfe_dim)
        x_gconvs, graph_feature_embedding = inputs
        batch_size = x_gconvs.shape[0]

        # projection
        gft = self.gft_projection(graph_feature_embedding)   # (P, 1, D)

        x_gconvs_graph = self.node_2_graph([tf.tile(gft[tf.newaxis, ...], [batch_size, 1, 1, 1]),
                                            x_gconvs]) + tf.tile(gft[tf.newaxis, ...], [batch_size, 1, 1, 1])
        x_gconvs_graph = tf.squeeze(x_gconvs_graph)  # (B, P, D)

        x_gconvs_graph_att = self.multi_graph_attention_fusion(x_gconvs_graph) + x_gconvs_graph  # (B, P, D)

        x_gconvs_node_att = self.graph_2_node([x_gconvs, x_gconvs_graph_att[..., tf.newaxis, :]]) + x_gconvs
        return x_gconvs_node_att                     # (B, P, N, D)


# Attention-Driven Dynamic Adaptive Graph Convolutional Network class
class ADAGCNCell(tf.keras.Model):
    def __init__(self, closeness_len, target_len, cheb_k, agcn_input_dim, agcn_output_dim, embed_dim,
                 heads, head_dim, dropout, num_node, closeness_use_dynamic, trend_use_dynamic, period_use_dynamic):
        super().__init__()
        self.closeness_len = closeness_len
        self.target_len = target_len
        self.closeness_use_dynamic = closeness_use_dynamic
        self.trend_use_dynamic = trend_use_dynamic
        self.period_use_dynamic = period_use_dynamic

        if self.closeness_use_dynamic:
            self.num_closeness_units = self.closeness_len

            self.closeness_agcn_sets = []
            for _ in range(self.num_closeness_units):
                self.closeness_agcn_sets.append(AGCN(cheb_k, agcn_input_dim, agcn_output_dim, embed_dim, num_node))
        else:
            self.num_closeness_units = 1

            self.closeness_agcn_sets = []
            for _ in range(self.num_closeness_units):
                self.closeness_agcn_sets.append(AGCN(cheb_k, agcn_input_dim, agcn_output_dim, embed_dim, num_node))

        if self.trend_use_dynamic:
            self.num_trend_units = self.target_len

            self.trend_agcn_sets = []
            for _ in range(self.num_trend_units):
                self.trend_agcn_sets.append(AGCN(cheb_k, agcn_input_dim, agcn_output_dim, embed_dim, num_node))
        else:
            self.num_trend_units = 1

            self.trend_agcn_sets = []
            for _ in range(self.num_trend_units):
                self.trend_agcn_sets.append(AGCN(cheb_k, agcn_input_dim, agcn_output_dim, embed_dim, num_node))

        if self.period_use_dynamic:
            self.num_period_units = self.target_len

            self.period_agcn_sets = []
            for _ in range(self.num_period_units):
                self.period_agcn_sets.append(AGCN(cheb_k, agcn_input_dim, agcn_output_dim, embed_dim, num_node))
        else:
            self.num_period_units = 1

            self.period_agcn_sets = []
            for _ in range(self.num_period_units):
                self.period_agcn_sets.append(AGCN(cheb_k, agcn_input_dim, agcn_output_dim, embed_dim, num_node))

        # Dynamic Multi-Graph Attention Fusion
        self.dynamic_multi_graph_attention_fusion = DMAF(agcn_output_dim, heads, head_dim, dropout)

    def call(self, inputs, *args, **kwargs):
        # inputs: [x_flow, closeness_node_embeds, trend_node_embeds, period_node_embeds, graph_feature_embedding]
        # x_flow: (B, N, P);
        # closeness_node_embeds: ((P, )N, dynamic_node_embed_dim + static_node_embed_dim);
        # trend_node_embeds: ((Q, )N, dynamic_node_embed_dim + static_node_embed_dim);
        # period_node_embeds: ((Q, )N, dynamic_node_embed_dim + static_node_embed_dim);
        # graph_feature_embedding: (P + 2Q, 1, gfe_dim)
        # output: (B, N, P, agcn_output_dim)

        x_flow, closeness_node_embeds, trend_node_embeds, period_node_embeds, graph_feature_embeds = inputs

        period_start = 0
        trend_start = self.target_len
        closeness_start = self.target_len * 2

        x_gconv_period_sets = []
        for i in range(self.target_len):
            if self.period_use_dynamic:
                x_gconv = self.period_agcn_sets[i](
                    [x_flow[..., period_start + i: period_start + (i + 1)], period_node_embeds[i]]
                )   # (B, N, D)
            else:
                x_gconv = self.period_agcn_sets[0](
                    [x_flow[..., period_start + i: period_start + (i + 1)], period_node_embeds]
                )   # (B, N, D)
            x_gconv_period_sets.append(x_gconv)
        x_gconvs_period = tf.stack(x_gconv_period_sets, axis=1)  # (B, Q, N, D)

        x_gconv_trend_sets = []
        for i in range(self.target_len):
            if self.trend_use_dynamic:
                x_gconv = self.trend_agcn_sets[i](
                    [x_flow[..., trend_start + i: trend_start + (i + 1)], trend_node_embeds[i]]
                )   # (B, N, D)
            else:
                x_gconv = self.trend_agcn_sets[0](
                    [x_flow[..., trend_start + i: trend_start + (i + 1)], trend_node_embeds]
                )   # (B, N, D)
            x_gconv_trend_sets.append(x_gconv)
        x_gconvs_trend = tf.stack(x_gconv_trend_sets, axis=1)  # (B, Q, N, D)

        x_gconv_closeness_sets = []
        for i in range(self.closeness_len):
            if self.closeness_use_dynamic:
                x_gconv = self.closeness_agcn_sets[i](
                    [x_flow[..., closeness_start + i: closeness_start + (i + 1)], closeness_node_embeds[i]]
                )   # (B, N, D)
            else:
                x_gconv = self.closeness_agcn_sets[0](
                    [x_flow[..., closeness_start + i: closeness_start + (i + 1)], closeness_node_embeds]
                )   # (B, N, D)
            x_gconv_closeness_sets.append(x_gconv)
        x_gconvs_closeness = tf.stack(x_gconv_closeness_sets, axis=1)  # (B, P, N, D)

        x_gconvs = tf.concat([x_gconvs_period, x_gconvs_trend, x_gconvs_closeness], axis=1)  # (B, 2Q + P, N, D)

        # extract the mutual relations between each dynamic graph spatial dependency based on graph feature embedding
        x_gconvs_atts = self.dynamic_multi_graph_attention_fusion([x_gconvs, graph_feature_embeds])
        x_gconvs_atts = tf.transpose(x_gconvs_atts, (0, 2, 1, 3))   # (B, N, 2Q + P, agcn_output_dim)

        return x_gconvs_atts                          # (B, N, 2Q + P, agcn_output_dim)


if __name__ == '__main__':
    pass

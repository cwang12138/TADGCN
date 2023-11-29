"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2023/9/10 21:00
# @Author  : Chen Wang
# @Site    : 
# @File    : tjmte_cell.py
# @Email   : chen.wang@ahnu.edu.cn
# @details : Time-Aware Joint Multi-View Temporal Encoder (TJMTE)
*****************************************************
"""
import tensorflow as tf
from einops import rearrange, repeat


# Residual Connection class
class Residual(tf.keras.Model):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def call(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# Pre-LayerNormalization class
class PreNorm(tf.keras.Model):
    def __init__(self, fn):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization()
        self.fn = fn

    def call(self, x, *args, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# FeedForward class
class FeedForward(tf.keras.Model):
    def __init__(self, out_dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(out_dim),
            tf.keras.layers.Dropout(dropout)
        ])

    def call(self, x, *args, **kwargs):
        return self.net(x)


# Time-Aware Periodicity Joint Attention class
class TimeAwarePeriodicityJointAttention(tf.keras.Model):
    def __init__(self, dimension, seq_out, heads=8, head_dim=64, dropout=0., is_multi_view=True):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dimension)

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.is_multi_view = is_multi_view
        self.norm = tf.keras.layers.LayerNormalization()
        self.to_qkv = tf.keras.layers.Dense(inner_dim * 3, use_bias=False)
        self.to_out = tf.keras.Sequential([
            tf.keras.layers.Dense(dimension),
            tf.keras.layers.Dropout(dropout)
        ]) if project_out else tf.identity

        self.period_weight = tf.Variable(tf.ones([1], tf.float32))
        self.trend_weights = tf.Variable(tf.ones([2], tf.float32))
        self.closeness_weight = tf.Variable(tf.ones([1], tf.float32))
        self.k = tf.Variable(tf.ones([1], tf.float32))
        self.c = tf.Variable(tf.ones([1], tf.float32))

        self.predictor = tf.keras.layers.Conv2D(seq_out, (1, dimension), use_bias=True)
        self.to_ta_q = tf.keras.layers.Dense(inner_dim, use_bias=False)
        self.to_ta_kv = tf.keras.layers.Dense(inner_dim * 2, use_bias=False)
        self.similarity_weights = tf.Variable(tf.ones([2], tf.float32))
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, *args, **kwargs):
        # inputs: [x_flow, x_time, y_time]
        # x_flow shape: (B, N, P, dimension)
        # x_time shape: (B, P, 5)
        # y_time shape: (B, Q, 5)
        x_flow, x_time, y_time = inputs

        # pre-norm
        x_flow = self.norm(x_flow)

        # time-aware matrix generation
        seq_len = y_time.shape[1]
        if self.is_multi_view:
            x_period = x_time[:, 0: seq_len]
            x_trend = x_time[:, seq_len: seq_len * 2]
            x_closeness = x_time[:, seq_len * 2: seq_len * 3]
            # period
            period_step_dist = self.time_step_dist_calc(x_period, y_time)
            similarity_dist_period = self.c / (self.k * tf.math.log(period_step_dist + 2.))
            time_aware_period = tf.multiply(self.period_weight, similarity_dist_period)  # (B, Q, P/3)
            # trend
            trend_step_dist = self.time_step_dist_calc(x_trend, y_time)
            trend_pattern = tf.reduce_sum(tf.cast(tf.equal(x_trend[:, 0, 1: 3], y_time[:, 0, 1: 3]), tf.int32), axis=-1)
            trend_weights_projection = tf.gather(self.trend_weights, trend_pattern)  # (B, )
            similarity_dist_trend = self.c / (self.k * tf.math.log(trend_step_dist + 2.))
            time_aware_trend = tf.multiply(trend_weights_projection[..., tf.newaxis, tf.newaxis], similarity_dist_trend)
            # (B, Q, P/3)
            # closeness
            closeness_step_dist = self.time_step_dist_calc(x_closeness, y_time)
            similarity_dist_closeness = self.c / (self.k * tf.math.log(closeness_step_dist + 2.))
            time_aware_closeness = tf.multiply(self.closeness_weight, similarity_dist_closeness)  # (B, Q, P/3)
            time_aware_matrix = tf.concat([time_aware_period, time_aware_trend, time_aware_closeness],
                                          axis=-1)  # (B, Q, P)
        else:
            x_closeness = x_time[:, 0: seq_len]
            closeness_step_dist = self.time_step_dist_calc(x_closeness, y_time)
            similarity_dist_closeness = self.c / (self.k * tf.math.log(closeness_step_dist + 2))
            time_aware_closeness = tf.multiply(self.closeness_weight, similarity_dist_closeness)
            time_aware_matrix = time_aware_closeness  # (B, Q, P)

        qkv = tf.split(self.to_qkv(x_flow), axis=-1, num_or_size_splits=3)  # (B, N, P, head_dim * heads)
        q, k, v = map(lambda t: rearrange(t, 'b n p (h d) -> b h n p d', h=self.heads), qkv)

        dots = tf.einsum('b h n i d, b h n j d -> b h n i j', q, k) * self.scale  # (B, H, N, P, P)
        attention = tf.nn.softmax(dots, axis=-1)

        out = tf.einsum('b h n i j, b h n j d -> b h n i d', attention, v)
        out = rearrange(out, 'b h n i d -> b n i (h d)')
        out = self.to_out(out)  # (B, N, P, tjmte_dim)

        temp_output = self.predictor(tf.transpose(out, (0, 1, 3, 2)))  # (B, N, 1, Q)
        ta_target_q_h = self.to_ta_q(tf.transpose(temp_output, (0, 1, 3, 2)))  # (B, N, Q, D * H)
        ta_target_q = rearrange(ta_target_q_h, 'b n q (h d) -> b h n q d', h=self.heads)
        ta_source_kv_h = tf.split(self.to_ta_kv(x_flow), axis=-1, num_or_size_splits=2)  # (B, N, Q, D * H)
        ta_source_k, ta_source_v = map(lambda t: rearrange(t, 'b n p (h d) -> b h n p d', h=self.heads), ta_source_kv_h)

        similarity_dots = tf.einsum('b h n i d, b h n j d -> b h n i j', ta_target_q, ta_source_k) * self.scale
        similarity = similarity_dots * self.similarity_weights[0] + \
                     (time_aware_matrix * self.similarity_weights[1])[:, tf.newaxis, tf.newaxis]
        similarity_att = tf.nn.softmax(similarity, axis=-1)
        current_output = tf.einsum('b h n i j, b h n j d -> b h n i d', similarity_att, ta_source_v)
        current_output = rearrange(current_output, 'b h n q d -> b n q (h d)')
        current_output = tf.squeeze(self.output_layer(current_output))  # (B, N, Q)

        return [out + x_flow, current_output + tf.squeeze(temp_output)]  # residual

    @staticmethod
    def time_step_dist_calc(time_x, time_y):
        time_x = tf.expand_dims(time_x[..., 3:], axis=1)  # (B, 1, P, D)
        time_y = tf.expand_dims(time_y[..., 3:], axis=2)  # (B, Q, 1, D)
        minutes_dist = tf.abs((time_y - time_x)[..., 0] * 60 + (time_y - time_x)[..., 1])  # (B, Q, P)
        steps_dist = tf.where(minutes_dist > 720, 1440 - minutes_dist, minutes_dist) // 5
        return tf.cast(steps_dist, tf.float32)  # (B, Q, P)


# Time-Aware Joint Multi-View Temporal Encoder class
class TJMTECell(tf.keras.Model):
    def __init__(self, dimension, seq_in, seq_out, heads, head_dim, dropout=0., is_multi_view=True):
        super().__init__()
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.JMVTE = []
        self.input_projection = tf.keras.layers.Dense(dimension)
        self.JMVTE.append(TimeAwarePeriodicityJointAttention(
            dimension=dimension,
            seq_out=seq_out,
            heads=heads,
            head_dim=head_dim,
            dropout=dropout,
            is_multi_view=is_multi_view
        ))
        self.JMVTE.append(Residual(PreNorm(FeedForward(
            out_dim=dimension,
            hidden_dim=dimension * 2,
            dropout=dropout
        ))))

    def call(self, inputs, *args, **kwargs):
        # inputs: [x_flow, x_time, y_time]
        # x_flow: (B, P, agcn_output_dim + dynamic_time_embed_dim);
        # x_time: (B, P, 5);
        # y_time: (B, Q, 5)
        # output: [x_flow, current_output]
        # x_flow: (B, N, P, tjmte_dim);
        # current_output: (B, N, Q)

        x_flow, x_time, y_time = inputs
        x_flow = self.input_projection(x_flow)                            # (B, N, P, tjmte_dim)

        x_flow, current_output = self.JMVTE[0]([x_flow, x_time, y_time])  # (B, N, P, tjmte_dim)
        x_flow = self.JMVTE[1](x_flow)                                    # (B, N, P, tjmte_dim)

        return x_flow, current_output


if __name__ == '__main__':
    pass

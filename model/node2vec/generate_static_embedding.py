"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2023/9/13 20:55
# @Author  : Chen Wang
# @Site    : 
# @File    : generate_static_embedding.py
# @Email   : chen.wang@ahnu.edu.cn
# @details : 
*****************************************************
"""
import node2vec
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import pandas as pd

is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 32
window_size = 10
iter = 1000


def convert_to_graph(sensor_ids_file, distances_file):
    """
    convert distance matrix to networkx graph.
    :param sensor_ids_file: sensor index file path, txt
    :param distances_file: distances file path, csv
    :return: networkx graph based on distances
    """
    # Load adj data
    with open(sensor_ids_file) as f:
        sensor_ids = f.read().strip().split(',')
    distance_df = pd.read_csv(distances_file, dtype={'from': 'str', 'to': 'str'})

    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):      # (Python >= 3.7, list and dict are ordered)
        sensor_id_to_ind[sensor_id] = i

    # Init directed graph object
    graph = nx.DiGraph()
    num_nodes = len(sensor_ids)

    # Add node to graph(Python >= 3.7)
    for node_idx in sensor_ids:
        graph.add_node(int(sensor_id_to_ind[node_idx]))

    # Add weighted edge to graph
    for row in distance_df.values:
        graph.add_edge(int(sensor_id_to_ind[row[0]]), int(sensor_id_to_ind[row[1]]), weight=row[2])

    return graph


def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size=dimensions, window=10, min_count=0, sg=1, workers=8, epochs=iter
    )
    model.wv.save_word2vec_format(output_file)


def process(sensor_ids_filepath, distances_filepath, SE_file):
    nx_G = convert_to_graph(sensor_ids_filepath, distances_filepath)
    G = node2vec.Graph(nx_G, is_directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    learn_embeddings(walks, dimensions, SE_file)


if __name__ == '__main__':
    for dataset in ['PEMS04', 'PEMS08']:
        print(f"Start {dataset} process!")
        sensor_ids_filepath = f'../../data/{dataset}/sensor_id_{dataset}.txt'
        distances_filepath = f'../../data/{dataset}/{dataset}.csv'
        SE_file = f'../../data/{dataset}/SE_{dataset}_{dimensions}.txt'
        process(sensor_ids_filepath, distances_filepath, SE_file)
        print(f"End {dataset} process!")

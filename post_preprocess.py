# coding=utf-8
# 2021.03 deleted  (1) count_accuracy, plot_solution; 
#                  (2) checkpoint_after_training
# Huawei Technologies Co., Ltd. 
# 
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Copyright (c) Ignavier Ng (https://github.com/ignavier/golem)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import networkx as nx
def is_dag(B):
    """
    Check whether B corresponds to a DAG.

    Parameters
    ----------
    B: numpy.ndarray
        [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))


def threshold_till_dag(B):
    """
    Remove the edges with smallest absolute weight until a DAG is obtained.

    Parameters
    ----------
    B: numpy.ndarray
        [d, d] weighted matrix.

    Return
    ------
    B: numpy.ndarray
        [d, d] weighted matrix of DAG.
    dag_thres: float
        Minimum threshold to obtain DAG.
    """
    if is_dag(B):
        return B, 0

    B = np.copy(B)
    # Get the indices with non-zero weight
    nonzero_indices = np.where(B != 0)
    # Each element in the list is a tuple (weight, j, i)
    weight_indices_ls = list(zip(B[nonzero_indices],
                                 nonzero_indices[0],
                                 nonzero_indices[1]))
    # Sort based on absolute weight
    sorted_weight_indices_ls = sorted(weight_indices_ls, key=lambda tup: abs(tup[0]))

    for weight, j, i in sorted_weight_indices_ls:
        if is_dag(B):
            # A DAG is found
            break

        # Remove edge with smallest absolute weight
        B[j, i] = 0
        dag_thres = abs(weight)

    return B, dag_thres


def postprocess(B):
    """
    Post-process estimated solution:
        (1) Thresholding.
        (2) Remove the edges with smallest absolute weight until a DAG
            is obtained.

    Parameters
    ----------
    B: numpy.ndarray
        [d, d] weighted matrix.
    graph_thres: float
        Threshold for weighted matrix. Default: 0.3.

    Return
    ------
    B: numpy.ndarray
        [d, d] weighted matrix of DAG.
    """
    B = np.copy(B)
    # Get the indices with non-zero weight
    ids = np.where(B != 0)
    # Each element in the list is a tuple (weight, j, i)
    weight_ids = list(zip(B[ids],ids[0],ids[1]))
    sorted_weight_ids = sorted(weight_ids, key=lambda tup: abs(tup[0]))
    weight_abs = (abs(sorted_weight_ids[-1][0]) - abs(sorted_weight_ids[0][0]))*0.10
    thres_num = abs(sorted_weight_ids[0][0]) + weight_abs
    B[np.abs(B) <= thres_num] = 0         #Thresholding-new
    # B[np.abs(B) <= graph_thres] = 0    # Thresholding-old
    # for i in range(len(B)):
    #     for j in range(i):
    #         if B[i,j] != 0 and B[j,i] != 0:
    #             B[i,j] = 0
    # B, _ = threshold_till_dag(B)

    return B


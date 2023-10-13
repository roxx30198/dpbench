# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

# import numpy as np
# import numpy.random as rand

# MAX_INIT_EDGES = 4
# MIN_EDGES = 2


# def initialize(numNodes):
#     visited = [False] * numNodes
#     graph_mask = [False] * numNodes
#     updating_graph_mask = [False] * numNodes
#     node_weight = [[0, 0]] * numNodes
#     node_edges = [0] * numNodes

#     for i in range(numNodes):
#         numEdges = (
#             rand.randint(1, 100) % (MAX_INIT_EDGES - MIN_EDGES + 1) + MIN_EDGES
#         )
#         for j in range(numEdges):
#             nodeId = rand.randint(0, numNodes)
#             weight = (
#                 rand.randint(1, 100) % (MAX_INIT_EDGES - MIN_EDGES + 1)
#                 + MIN_EDGES
#             )
#             node_weight[i][0] = nodeId
#             node_weight[i][1]

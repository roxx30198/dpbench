// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>

struct Node
{
    int starting;
    int no_of_edges;
};

void Kernel(Node *g_graph_nodes,
            int *g_graph_edges,
            bool *g_graph_mask,
            bool *g_updating_graph_mask,
            bool *g_graph_visited,
            int *g_cost,
            int no_of_nodes,
            sycl::nd_item<3> item_ct1,
            int max_blocks)
{
    int MAX_THREADS_PER_BLOCK = max_blocks;
    int tid = item_ct1.get_group(2) * MAX_THREADS_PER_BLOCK +
              item_ct1.get_local_id(2);
    if (tid < no_of_nodes && g_graph_mask[tid]) {
        g_graph_mask[tid] = false;
        for (int i = g_graph_nodes[tid].starting;
             i < (g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting);
             i++)
        {
            int id = g_graph_edges[i];
            if (!g_graph_visited[id]) {
                g_cost[id] = g_cost[tid] + 1;
                g_updating_graph_mask[id] = true;
            }
        }
    }
}

void Kernel2(bool *g_graph_mask,
             bool *g_updating_graph_mask,
             bool *g_graph_visited,
             bool *g_over,
             int no_of_nodes,
             sycl::nd_item<3> item_ct1,
             int max_blocks)
{
    int MAX_THREADS_PER_BLOCK = max_blocks;
    int tid = item_ct1.get_group(2) * MAX_THREADS_PER_BLOCK +
              item_ct1.get_local_id(2);
    if (tid < no_of_nodes && g_updating_graph_mask[tid]) {

        g_graph_mask[tid] = true;
        g_graph_visited[tid] = true;
        *g_over = true;
        g_updating_graph_mask[tid] = false;
    }
}

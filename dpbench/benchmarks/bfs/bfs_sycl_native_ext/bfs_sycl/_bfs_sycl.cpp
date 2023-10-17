// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "_bfs_kernel.hpp"
#include <dpctl4pybind11.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int no_of_nodes;
int edge_list_size;

void BFSGraph()
{
    sycl::queue q_ct1{selector};

    int source = 0;

    int num_of_blocks = 1;
    int num_of_threads_per_block = no_of_nodes;
    int MAX_THREADS_PER_BLOCK =
        q_ct1.get_device()
            .get_info<cl::sycl::info::device::max_work_group_size>();

    // Make execution Parameters according to the number of nodes
    // Distribute threads across multiple Blocks if necessary
    if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
        num_of_blocks = (int)ceil(no_of_nodes / (double)MAX_THREADS_PER_BLOCK);
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
    }

    // allocate host memory
    Node *h_graph_nodes = (Node *)malloc(sizeof(Node) * no_of_nodes);
    bool *h_graph_mask = (bool *)malloc(sizeof(bool) * no_of_nodes);
    bool *h_updating_graph_mask = (bool *)malloc(sizeof(bool) * no_of_nodes);
    bool *h_graph_visited = (bool *)malloc(sizeof(bool) * no_of_nodes);

    int start, edgeno;
    source = 0;

    // set the source node as true in the mask
    h_graph_mask[source] = true;
    h_graph_visited[source] = true;

    int id, cost;
    int *h_graph_edges = (int *)malloc(sizeof(int) * edge_list_size);
    for (int i = 0; i < edge_list_size; i++) {
        fscanf(fp, "%d", &id);
        fscanf(fp, "%d", &cost);
        h_graph_edges[i] = id;
    }

    Node *d_graph_nodes;
    int *d_graph_edges;
    bool *d_graph_mask;
    bool *d_updating_graph_mask;
    bool *d_graph_visited;
    int *d_cost;
    bool *d_over;

    // allocate mem for the result on host side
    int *h_cost = (int *)malloc(sizeof(int) * no_of_nodes);
    for (int i = 0; i < no_of_nodes; i++)
        h_cost[i] = -1;
    h_cost[source] = 0;

    // setup execution parameters
    sycl::range<3> grid(1, 1, num_of_blocks);
    sycl::range<3> threads(1, 1, num_of_threads_per_block);
    int k = 0;

    d_graph_nodes = sycl::malloc_device<Node>(no_of_nodes, q_ct1);
    d_graph_edges = sycl::malloc_device<int>(edge_list_size, q_ct1);
    d_graph_mask = sycl::malloc_device<bool>(no_of_nodes, q_ct1);
    d_updating_graph_mask = sycl::malloc_device<bool>(no_of_nodes, q_ct1);
    d_graph_visited = sycl::malloc_device<bool>(no_of_nodes, q_ct1);
    // allocate device memory for result
    d_cost = sycl::malloc_device<int>(no_of_nodes, q_ct1);
    // make a bool to check if the execution is over
    d_over = sycl::malloc_device<bool>(1, q_ct1);

    // Copy the Node list to device memory
    q_ct1.memcpy(d_graph_nodes, h_graph_nodes, sizeof(Node) * no_of_nodes)
        .wait();

    // Copy the Edge List to device Memory
    q_ct1.memcpy(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size)
        .wait();

    // Copy the Mask to device memory
    q_ct1.memcpy(d_graph_mask, h_graph_mask, sizeof(bool) * no_of_nodes).wait();

    q_ct1
        .memcpy(d_updating_graph_mask, h_updating_graph_mask,
                sizeof(bool) * no_of_nodes)
        .wait();

    // Copy the Visited nodes array to device memory
    q_ct1.memcpy(d_graph_visited, h_graph_visited, sizeof(bool) * no_of_nodes)
        .wait();

    q_ct1.memcpy(d_cost, h_cost, sizeof(int) * no_of_nodes).wait();

    bool stop;
    // Call the Kernel untill all the elements of Frontier are not false
    do {
        // if no thread changes this value then the loop stops
        stop = false;
        q_ct1.memcpy(d_over, &stop, sizeof(bool)).wait();
        /*
        DPCT1049:0: The workgroup size passed to the SYCL
         * kernel may exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the workgroup size
         * if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            auto no_of_nodes_ct6 = no_of_nodes;

            cgh.parallel_for(
                sycl::nd_range<3>(grid * threads, threads),
                [=](sycl::nd_item<3> item_ct1) {
                    Kernel(d_graph_nodes, d_graph_edges, d_graph_mask,
                           d_updating_graph_mask, d_graph_visited, d_cost,
                           no_of_nodes_ct6, item_ct1, MAX_THREADS_PER_BLOCK);
                });
        });
        // check if kernel execution generated and error

        /*
        DPCT1049:1: The workgroup size passed to the SYCL
         * kernel may exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the workgroup size
         * if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            auto no_of_nodes_ct4 = no_of_nodes;

            cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                             [=](sycl::nd_item<3> item_ct1) {
                                 Kernel2(d_graph_mask, d_updating_graph_mask,
                                         d_graph_visited, d_over,
                                         no_of_nodes_ct4, item_ct1,
                                         MAX_THREADS_PER_BLOCK);
                             });
        });
        // check if kernel execution generated and error

        q_ct1.memcpy(&stop, d_over, sizeof(bool)).wait();
        k++;
    } while (stop);

    // copy result from device to host
    q_ct1.memcpy(h_cost, d_cost, sizeof(int) * no_of_nodes).wait();
    f

        sycl::free(d_graph_nodes, q_ct1);
    sycl::free(d_graph_edges, q_ct1);
    sycl::free(d_graph_mask, q_ct1);
    sycl::free(d_updating_graph_mask, q_ct1);
    sycl::free(d_graph_visited, q_ct1);
    sycl::free(d_cost, q_ct1);

    // Store the result into a file
    FILE *fpo = fopen("result.txt", "w");
    for (int i = 0; i < no_of_nodes; i++)
        fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
    fclose(fpo);
    printf("Result stored in result.txt\n");

    // cleanup memory
    free(h_graph_nodes);
    free(h_graph_edges);
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);
    free(h_cost);
}

PYBIND11_MODULE(_bfs_sycl, m)
{
    // Import the dpctl extensions
    import_dpctl();

    m.def("bfs", &bfs_sync, "DPC++ implementation of the pathfinder",
          py::arg("data"), py::arg("rows"), py::arg("cols"),
          py::arg("pyramid_height"), py::arg("result"));
}

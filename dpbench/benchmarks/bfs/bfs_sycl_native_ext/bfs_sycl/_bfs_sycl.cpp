/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology -
 Hyderabad. All rights reserved.

  Permission to use, copy, modify and distribute this software and its
 documentation for educational purpose is hereby granted without fee, provided
 that the above copyright notice and this permission notice appear in all copies
 of this software and that you do not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS,
 IMPLIED OR OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <cuda.h>
#include "../common.hpp"
#include "bfs.hpp"

#ifdef TIME_IT
#include <sys/time.h>
#endif

int no_of_nodes;
int edge_list_size;
FILE *fp;

void BFSGraph(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    no_of_nodes = 0;
    edge_list_size = 0;
    BFSGraph(argc, argv);
}

void Usage(int argc, char **argv)
{

    fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
}

#ifdef TIME_IT
long long get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int argc, char **argv)
{
    // dpct::device_ext &dev_ct1 = dpct::get_current_device();
    // sycl::queue &q_ct1 = dev_ct1.default_queue();
#ifdef TIME_IT
    long long time0;
    long long time1;
    long long time2;
    long long time3;
    long long time4;
    long long time5;
    long long time6;
#endif

    char *input_f;
    if (argc != 2) {
        Usage(argc, argv);
        exit(0);
    }

    input_f = argv[1];
    printf("Reading File\n");
    // Read in Graph from a file
    fp = fopen(input_f, "r");
    if (!fp) {
        printf("Error Reading graph file\n");
        return;
    }

#ifdef TIME_IT
    time0 = get_time();
#endif

#ifdef NVIDIA_GPU
    NvidiaGpuSelector selector{};
#elif INTEL_GPU
    IntelGpuSelector selector{};
#else
    sycl::cpu_selector selector{};
#endif

    sycl::queue q_ct1{selector};

#ifdef TIME_IT
    time1 = get_time();
#endif

    std::cout << "Running on "
              << q_ct1.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    int source = 0;

    fscanf(fp, "%d", &no_of_nodes);

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
    // initalize the memory
    for (unsigned int i = 0; i < no_of_nodes; i++) {
        fscanf(fp, "%d %d", &start, &edgeno);
        h_graph_nodes[i].starting = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        h_graph_mask[i] = false;
        h_updating_graph_mask[i] = false;
        h_graph_visited[i] = false;
    }

    // read the source node from the file
    fscanf(fp, "%d", &source);
    source = 0;

    // set the source node as true in the mask
    h_graph_mask[source] = true;
    h_graph_visited[source] = true;

    fscanf(fp, "%d", &edge_list_size);

    int id, cost;
    int *h_graph_edges = (int *)malloc(sizeof(int) * edge_list_size);
    for (int i = 0; i < edge_list_size; i++) {
        fscanf(fp, "%d", &id);
        fscanf(fp, "%d", &cost);
        h_graph_edges[i] = id;
    }

    if (fp)
        fclose(fp);

    printf("Read File\n");

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

#ifdef TIME_IT
    time2 = get_time();
#endif

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

#ifdef TIME_IT
    time3 = get_time();
#else
    printf("Copied Everything to GPU memory\n");

    printf("Start traversing the tree\n");
#endif
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

#ifdef TIME_IT
    time4 = get_time();
#else
    printf("Kernel Executed %d times\n", k);
#endif
    // copy result from device to host
    q_ct1.memcpy(h_cost, d_cost, sizeof(int) * no_of_nodes).wait();

#ifdef TIME_IT
    time5 = get_time();
#endif

    sycl::free(d_graph_nodes, q_ct1);
    sycl::free(d_graph_edges, q_ct1);
    sycl::free(d_graph_mask, q_ct1);
    sycl::free(d_updating_graph_mask, q_ct1);
    sycl::free(d_graph_visited, q_ct1);
    sycl::free(d_cost, q_ct1);

#ifdef TIME_IT
    time6 = get_time();
#endif

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

#ifdef TIME_IT
    printf("Time spent in different stages of GPU_CUDA KERNEL:\n");

    printf("%15.12f s, %15.12f % : GPU: SET DEVICE / DRIVER INIT\n",
           (float)(time1 - time0) / 1000000,
           (float)(time1 - time0) / (float)(time6 - time0) * 100);
    printf("%15.12f s, %15.12f % : GPU MEM: ALO\n",
           (float)(time2 - time1) / 1000000,
           (float)(time2 - time1) / (float)(time6 - time0) * 100);
    printf("%15.12f s, %15.12f % : GPU MEM: COPY IN\n",
           (float)(time3 - time2) / 1000000,
           (float)(time3 - time2) / (float)(time6 - time0) * 100);

    printf("%15.12f s, %15.12f % : GPU: KERNEL\n",
           (float)(time4 - time3) / 1000000,
           (float)(time4 - time3) / (float)(time6 - time0) * 100);

    printf("%15.12f s, %15.12f % : GPU MEM: COPY OUT\n",
           (float)(time5 - time4) / 1000000,
           (float)(time5 - time4) / (float)(time6 - time0) * 100);
    printf("%15.12f s, %15.12f % : GPU MEM: FRE\n",
           (float)(time6 - time5) / 1000000,
           (float)(time6 - time5) / (float)(time6 - time0) * 100);

    printf("Total time:\n");
    printf("%.12f s\n", (float)(time6 - time0) / 1000000);
#endif
}

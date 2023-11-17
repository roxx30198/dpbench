// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "_hotspot_kernel.hpp"
#include <dpctl4pybind11.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD (3.0e6)
/* required precision in degrees	*/
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP 0.5
#define EXPAND_RATE 2

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

template <typename... Args> bool ensure_compatibility(const Args &...args)
{
    std::vector<dpctl::tensor::usm_ndarray> arrays = {args...};

    auto arr = arrays.at(0);

    for (auto &arr : arrays) {
        if (!(arr.get_flags() & (USM_ARRAY_C_CONTIGUOUS))) {
            std::cerr << "All arrays need to be C contiguous.\n";
            return false;
        }
    }
    return true;
}

void hotspot_sync(dpctl::tensor::usm_ndarray temperature,
                  dpctl::tensor::usm_ndarray power,
                  int rows,
                  int cols,
                  dpctl::tensor::usm_ndarray result)
{
    sycl::queue q_ct1;
    int size;

    int total_iterations = 60;
    int pyramid_height = 1; // number of iterations

    size = rows * cols;

    if (!ensure_compatibility(temperature, power, result))
        throw std::runtime_error("Input arrays are not acceptable.");

    /* --------------- pyramid parameters --------------- */

    int borderCols = (pyramid_height)*EXPAND_RATE / 2;
    int borderRows = (pyramid_height)*EXPAND_RATE / 2;
    int smallBlockCol = BLOCK_SIZE - (pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE - (pyramid_height)*EXPAND_RATE;
    int blockCols =
        cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);
    int blockRows =
        rows / smallBlockRow + ((rows % smallBlockRow == 0) ? 0 : 1);

    auto temp_value = temperature.get_data<float>();
    auto power_value = power.get_data<float>();

    float *MatrixTemp[2], *MatrixPower;

    MatrixTemp[0] = sycl::malloc_device<float>(size, q_ct1);
    MatrixTemp[1] = sycl::malloc_device<float>(size, q_ct1);

    q_ct1.memcpy(MatrixTemp[0], temp_value, sizeof(float) * size).wait();

    MatrixPower = sycl::malloc_device<float>(size, q_ct1);

    q_ct1.memcpy(MatrixPower, power_value, sizeof(float) * size).wait();

    sycl::range<3> dimBlock(1, BLOCK_SIZE, BLOCK_SIZE);
    sycl::range<3> dimGrid(1, blockRows, blockCols);

    float grid_height = chip_height / rows;
    float grid_width = chip_width / cols;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;
    float t;

    int src = 1, dst = 0;

    for (t = 0; t < total_iterations; t += pyramid_height) {
        int temp = src;
        src = dst;
        dst = temp;
        /*
        DPCT1049:0: The workgroup size passed to the SYCL
         * kernel may exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the workgroup size if
         * needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::range<2> temp_range_ct1(BLOCK_SIZE, BLOCK_SIZE);
            sycl::range<2> power_range_ct1(BLOCK_SIZE, BLOCK_SIZE);
            sycl::range<2> temp_t_range_ct1(BLOCK_SIZE, BLOCK_SIZE);

            sycl::local_accessor<float, 2> temp_acc_ct1(temp_range_ct1, cgh);
            sycl::local_accessor<float, 2> power_acc_ct1(power_range_ct1, cgh);
            sycl::local_accessor<float, 2> temp_t_acc_ct1(temp_t_range_ct1,
                                                          cgh);

            auto MatrixTemp_src_ct2 = MatrixTemp[src];
            auto MatrixTemp_dst_ct3 = MatrixTemp[dst];

            cgh.parallel_for(
                sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                [=](sycl::nd_item<3> item_ct1) {
                    calculate_temp(MIN(pyramid_height, total_iterations - t),
                                   MatrixPower, MatrixTemp_src_ct2,
                                   MatrixTemp_dst_ct3, cols, rows, borderCols,
                                   borderRows, Cap, Rx, Ry, Rz, step, item_ct1,
                                   temp_acc_ct1, power_acc_ct1, temp_t_acc_ct1);
                });
        });
    }
    auto result_value = result.get_data<float>();
    q_ct1.memcpy(result_value, MatrixTemp[dst], sizeof(float) * size).wait();

    for (int i = 0; i < 16; i++)
        std::cout << result_value[i] << " ";
    std::cout << "\n";

    sycl::free(MatrixTemp[0], q_ct1);
    sycl::free(MatrixTemp[1], q_ct1);
}

PYBIND11_MODULE(_hotspot_sycl, m)
{
    // Import the dpctl extensions
    import_dpctl();

    m.def("hotspot", &hotspot_sync, "DPC++ implementation of the hotspot",
          py::arg("temperature"), py::arg("power"), py::arg("rows"),
          py::arg("cols"), py::arg("result"));
}

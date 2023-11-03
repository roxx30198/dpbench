// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "_nw_kernel.hpp"
#include <dpctl4pybind11.hpp>

template <typename... Args> bool ensure_compatibility(const Args &...args)
{
    std::vector<dpctl::tensor::usm_ndarray> arrays = {args...};

    auto arr = arrays.at(0);
    auto q = arr.get_queue();
    auto type_flag = arr.get_typenum();
    auto arr_size = arr.get_size();

    for (auto &arr : arrays) {
        if (!(arr.get_flags() & (USM_ARRAY_C_CONTIGUOUS))) {
            std::cerr << "All arrays need to be C contiguous.\n";
            return false;
        }
        if (arr.get_typenum() != type_flag) {
            std::cerr << "All arrays should be of same elemental type.\n";
            return false;
        }
        if (arr.get_ndim() > 1) {
            std::cerr << "All arrays expected to be single-dimensional.\n";
            return false;
        }
    }
    return true;
}

void nw_sync(dpctl::tensor::usm_ndarray input_itemsets,
             dpctl::tensor::usm_ndarray reference,
             dpctl::tensor::usm_ndarray output_datasets,
             int max_rows,
             int max_cols,
             int penalty,
             dpctl::tensor::usm_ndarray result)
{
    if (!ensure_compatibility(input_itemsets, reference, result))
        throw std::runtime_error("Input arrays are not acceptable.");

    sycl::queue q_ct1;

    // the lengths of the two sequences should be able to divided by 16.
    // And at current stage  max_rows needs to equal max_col

    int size = max_cols * max_rows;

    sycl::range<3> dimGrid(1, 1, 1);
    sycl::range<3> dimBlock(1, 1, BLOCK_SIZE);
    int block_width = (max_cols - 1) / BLOCK_SIZE;

    // process top-left matrix
    auto input_value = input_itemsets.get_data<int>();
    auto reference_value = reference.get_data<int>();
    auto out_value = output_datasets.get_data<int>();
    auto result_value = result.get_data<int>();
    int c = 1;
    std::cout << "block_widhth " << block_width << "\n";
    for (int i = 1; i <= block_width; i++) {
        std::cout << "Inside kernel 1 " << c++ << "\n";
        dimGrid[2] = i;
        dimGrid[1] = 1;

        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::range<2> temp_range_ct1(BLOCK_SIZE + 1, BLOCK_SIZE + 1);
            sycl::range<2> ref_range_ct1(BLOCK_SIZE, BLOCK_SIZE);

            sycl::local_accessor<int, 2> temp_acc_ct1(temp_range_ct1, cgh);
            sycl::local_accessor<int, 2> ref_acc_ct1(ref_range_ct1, cgh);

            cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                             [=](sycl::nd_item<3> item_ct1) {
                                 needle_device_shared_1(
                                     reference_value, input_value, max_cols,
                                     penalty, i, block_width, item_ct1,
                                     temp_acc_ct1, ref_acc_ct1);
                             });
        });
    }
    c = 1;
    // process bottom-right matrix
    for (int i = block_width - 1; i >= 1; i--) {
        std::cout << "Inside kernel 2 " << c++ << "\n";

        dimGrid[2] = i;
        dimGrid[1] = 1;

        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::range<2> temp_range_ct1(BLOCK_SIZE + 1, BLOCK_SIZE + 1);
            sycl::range<2> ref_range_ct1(BLOCK_SIZE, BLOCK_SIZE);

            sycl::local_accessor<int, 2> temp_acc_ct1(temp_range_ct1, cgh);
            sycl::local_accessor<int, 2> ref_acc_ct1(ref_range_ct1, cgh);

            cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                             [=](sycl::nd_item<3> item_ct1) {
                                 needle_device_shared_2(
                                     reference_value, input_value, max_cols,
                                     penalty, i, block_width, item_ct1,
                                     temp_acc_ct1, ref_acc_ct1);
                             });
        });
    }
    q_ct1.memcpy(out_value, input_value, sizeof(int) * size).wait();

    for (int i = max_rows - 2, j = max_rows - 2, k = 0; i >= 0 && j >= 0;) {

        int nw = 0, n = 0, w = 0, traceback = 0;

        if (i > 0 && j > 0) {
            nw = out_value[(i - 1) * max_cols + j - 1];
            w = out_value[i * max_cols + j - 1];
            n = out_value[(i - 1) * max_cols + j];
        }
        else if (i == 0) {
            nw = n = LIMIT;
            w = out_value[i * max_cols + j - 1];
        }
        else {
            nw = w = LIMIT;
            n = out_value[(i - 1) * max_cols + j];
        }

        int new_nw, new_w, new_n;
        new_nw = nw + reference_value[i * max_cols + j];
        new_w = w - penalty;
        new_n = n - penalty;

        traceback = maximum(new_nw, new_w, new_n);

        if (traceback == new_nw)
            traceback = nw;
        if (traceback == new_w)
            traceback = w;
        if (traceback == new_n)
            traceback = n;

        result_value[k] = traceback;
        k++;

        if (traceback == nw) {
            i--;
            j--;
            continue;
        }
        else if (traceback == w) {
            j--;
            continue;
        }
        else {
            i--;
            continue;
        }
    }
    for (int i = 0; i < max_rows; i++) {
        for (int j = 0; j < max_cols; j++) {
            std::cout << result_value[i * max_cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n\n";
}

PYBIND11_MODULE(_nw_sycl, m)
{
    // Import the dpctl extensions
    import_dpctl();

    m.def("nw", &nw_sync, "DPC++ implementation of the needleman-wunsch",
          py::arg("input_itemsets"), py::arg("reference"),
          py::arg("output_datasets"), py::arg("max_rows"), py::arg("max_cols"),
          py::arg("penalty"), py::arg("result"));
}

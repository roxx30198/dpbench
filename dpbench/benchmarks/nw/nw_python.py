# # SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
# #
# # SPDX-License-Identifier: Apache-2.0

# import dpnp
# import numba_dpex

# BLOCK_SIZE = 16
# LIMIT = -999


# def maximum(a, b, c):
#     k = a if a > b else b
#     k = c if c > k else k
#     return k


# def neddle_kernel_1(reference, input_itemsets, cols, penalty, i, temp, ref):
#     index = (
#         cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (cols + 1)
#     )
#     index_n = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (1)
#     index_w = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + (cols)
#     index_nw = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x

#     if tx == 0:
#         temp[tx][0] = input_itemsets[index_nw]

#     for ty in range(BLOCK_SIZE):
#         ref[ty][tx] = reference[index + cols * ty]

#     temp[tx + 1][0] = input_itemsets[index_w + cols * tx]

#     temp[0][tx + 1] = input_itemsets[index_n]

#     for m in range(BLOCK_SIZE):
#         if tx <= m:
#             t_index_x = tx + 1
#             t_index_y = m - tx + 1
#             temp[t_index_y][t_index_x] = maximum(
#                 temp[t_index_y - 1][t_index_x - 1]
#                 + ref[t_index_y - 1][t_index_x - 1],
#                 temp[t_index_y][t_index_x - 1] - penalty,
#                 temp[t_index_y - 1][t_index_x] - penalty,
#             )

#     for m in range(BLOCK_SIZE - 2, -1, -1):
#         if tx <= m:
#             t_index_x = tx + BLOCK_SIZE - m
#             t_index_y = BLOCK_SIZE - tx
#             temp[t_index_y][t_index_x] = maximum(
#                 temp[t_index_y - 1][t_index_x - 1]
#                 + ref[t_index_y - 1][t_index_x - 1],
#                 temp[t_index_y][t_index_x - 1] - penalty,
#                 temp[t_index_y - 1][t_index_x] - penalty,
#             )

#     for ty in range(BLOCK_SIZE):
#         input_itemsets[index + ty * cols] = temp[ty + 1][tx + 1]


# def neddle_kernel_2(
#     reference, input_itemsets, cols, penalty, i, block_width, temp, ref
# ):
#     b_index_x = bx + block_width - i
#     b_index_y = block_width - bx - 1

#     index = (
#         cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (cols + 1)
#     )
#     index_n = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (1)
#     index_w = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + (cols)
#     index_nw = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x

#     for ty in range(BLOCK_SIZE):
#         ref[ty][tx] = reference[index + cols * ty]

#     if tx == 0:
#         temp[tx][0] = input_itemsets[index_nw]

#     temp[tx + 1][0] = input_itemsets[index_w + cols * tx]

#     temp[0][tx + 1] = input_itemsets[index_n]

#     for m in range(BLOCK_SIZE):
#         if tx <= m:
#             t_index_x = tx + 1
#             t_index_y = m - tx + 1
#             temp[t_index_y][t_index_x] = maximum(
#                 temp[t_index_y - 1][t_index_x - 1]
#                 + ref[t_index_y - 1][t_index_x - 1],
#                 temp[t_index_y][t_index_x - 1] - penalty,
#                 temp[t_index_y - 1][t_index_x] - penalty,
#             )

#     for m in range(BLOCK_SIZE - 2, -1, -1):
#         if tx <= m:
#             t_index_x = tx + BLOCK_SIZE - m
#             t_index_y = BLOCK_SIZE - tx
#             temp[t_index_y][t_index_x] = maximum(
#                 temp[t_index_y - 1][t_index_x - 1]
#                 + ref[t_index_y - 1][t_index_x - 1],
#                 temp[t_index_y][t_index_x - 1] - penalty,
#                 temp[t_index_y - 1][t_index_x] - penalty,
#             )

#     for ty in range(BLOCK_SIZE):
#         input_itemsets[index + ty * cols] = temp[ty + 1][tx + 1]


# def nw(
#     input_itemsets,
#     reference,
#     output_datasets,
#     max_rows,
#     max_cols,
#     penalty,
#     result,
# ):
#     # process top-left matrix
#     for i in range(1, block_width):
#         neddle_kernel_1[numba_dpex.NdRange(dimGrid, dimBlock)](
#             reference, input_itemsets, max_cols, penalty, i, temp, ref
#         )

#     # process bottom-right matrix
#     for i in range(block_width - 1, 0, -1):
#         neddle_kernel_2[numba_dpex.NdRange(dimGrid, dimBlock)](
#             reference,
#             input_itemsets,
#             max_cols,
#             penalty,
#             i,
#             block_width,
#             temp,
#             ref,
#         )

#     output_datasets = input_itemsets

#     i, j = max_rows - 2, max_rows - 2
#     k = 0
#     while True:
#         if i == 0 and j == 0:
#             break
#         nw, n, w, traceback = 0, 0, 0, 0

#         if i > 0 and j > 0:
#             nw = output_datasets[(i - 1) * max_cols + j - 1]
#             w = output_datasets[i * max_cols + j - 1]
#             n = output_datasets[(i - 1) * max_cols + j]

#         elif i == 0:
#             nw = n = LIMIT
#             w = output_datasets[i * max_cols + j - 1]

#         elif j == 0:
#             nw = w = LIMIT
#             n = output_datasets[(i - 1) * max_cols + j]

#         new_nw = nw + reference[i * max_cols + j]
#         new_w = w - penalty
#         new_n = n - penalty

#         traceback = maximum(new_nw, new_w, new_n)
#         result[k] = traceback
#         k += 1

#         if traceback == new_nw:
#             traceback = nw
#         if traceback == new_w:
#             traceback = w
#         if traceback == new_n:
#             traceback = n

#         if traceback == nw:
#             i -= 1
#             j -= 1
#             continue

#         elif traceback == w:
#             j -= 1
#             continue
#         elif traceback == n:
#             i -= 1
#             continue

# # SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
# #
# # SPDX-License-Identifier: Apache-2.0
"""Numba-dpex implementation for hotspot."""
# import dpnp
# import numba_dpex

# STR_SIZE = 256

# # maximum power density possible (say 300W for a 10mm x 10mm chip)
# MAX_PD = 3.0e6
# # required precision in degrees
# PRECISION = 0.001
# SPEC_HEAT_SI = 1.75e6
# K_SI = 100
# # capacitance fitting factor
# FACTOR_CHIP = 0.5

# # chip parameters
# t_chip = 0.0005
# chip_height = 0.016
# chip_width = 0.016
# BLOCK_SIZE = 16


# @numba_dpex.kernel(debug=True)
# def _hotspot_kernel(
#     iteration,
#     power,
#     temp_src,
#     temp_dest,
#     cols,
#     rows,
#     cap,
#     Rx,
#     Ry,
#     Rz,
#     step,
#     result,
# ):
#     amb_temp = 80.0
#     rx_1 = 1 / Rx
#     ry_1 = 1 / Ry
#     rz_1 = 1 / Rz

#     step_div_cap = step / cap

#     tx = numba_dpex.get_local_id(1)
#     ty = numba_dpex.get_local_id(0)

#     n = ty - 1
#     s = ty + 1
#     e = tx + 1
#     w = tx - 1

#     validXmax = cols - 1
#     validXmin = 0
#     validYmax = rows - 1
#     validYmin = 0

#     n = n if n >= 1 else validYmin
#     s = s if s < rows else validYmax
#     e = e if e < cols else validXmax
#     w = w if w >= 1 else validXmin

#     # convert to corresponding 1D address
#     cur_element = tx * cols + ty
#     n_ind = n * cols + ty
#     s_ind = s * cols + ty
#     e_ind = (tx - 1) * cols + e
#     w_ind = (tx - 1) * cols + w

#     for i in range(iteration):
#         temp_dest[cur_element] = temp_src[cur_element] + step_div_cap * (
#             power[cur_element]
#             + temp_src[s_ind]
#             + temp_src[n_ind]
#             - 2.0 * temp_src[cur_element] * ry_1
#             + temp_src[e_ind]
#             + temp_src[w_ind]
#             - 2.0 * temp_src[cur_element] * rx_1
#             + amb_temp
#             - temp_src[cur_element] * rz_1
#         )

#         numba_dpex.barrier(numba_dpex.LOCAL_MEM_FENCE)
#         temp_src[cur_element] = temp_dest[cur_element]
#         if i == iteration - 1:
#             break
#         numba_dpex.barrier(numba_dpex.LOCAL_MEM_FENCE)

#     result[cur_element] = temp_dest[cur_element]


# def hotspot(temperature, power, rows, cols, result):
#     grid_height = chip_height / rows
#     grid_width = chip_width / cols
#     Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height
#     Rx = grid_width / (2.0 * K_SI * t_chip * grid_height)
#     Ry = grid_height / (2.0 * K_SI * t_chip * grid_width)
#     Rz = t_chip / (K_SI * grid_height * grid_width)
#     max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI)
#     step = PRECISION / max_slope

#     total_iterations = 60
#     pyramid_height = 1
#     size = rows * cols

#     temp_src = dpnp.array(temperature, dtype=float)
#     temp_dest = dpnp.array([0] * size, dtype=float)

#     for t in range(0, total_iterations, pyramid_height):
#         iteration = min(pyramid_height, total_iterations - t)
#         _hotspot_kernel[numba_dpex.Range(rows, cols)](
#             iteration,
#             power,
#             temp_src,
#             temp_dest,
#             cols,
#             rows,
#             Cap,
#             Rx,
#             Ry,
#             Rz,
#             step,
#             result,
#         )

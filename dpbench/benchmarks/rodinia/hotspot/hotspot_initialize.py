# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
"""Initialization function for matrices for hotspot."""

# Make sure rows==cols(square matrices only)
# LOW = 320
# HIGH = 330


# def initialize(rows, cols, types_dict=None):
#     import numpy as np
#     import numpy.random as rnd

#     return (
#         rnd.randint(LOW, HIGH, (rows * cols)) + rnd.random(),
#         rnd.random(rows * cols),
#         np.empty((rows * cols), dtype=float),
#     )

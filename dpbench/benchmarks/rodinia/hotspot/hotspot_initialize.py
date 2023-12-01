# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
"""Initialization function for matrices for hotspot."""

# Make sure rows==cols(square matrices only)
LOW = 320
HIGH = 330


def initialize(rows, cols, types_dict=None):
    """Initialize the matrices based on size and type.

    Args:
        rows: number of rows.
        cols: number of cols.
        types_dict: data type of operand.

    Returns: temp: temperaturn matrix (rows x cols).
             power: power matrix (rows x cols).
             result: result of operation.
    """
    import numpy as np

    # import numpy.random as rnd
    dtype = types_dict["float"]
    return (
        np.random.randn((rows, cols), dtype=dtype),
        np.random.randn((rows, cols), dtype=dtype),
        np.array((rows, cols), dtype=dtype),
    )
    # return (
    #     np.array(
    #         [
    #             323.865780,
    #             323.898699,
    #             323.944688,
    #             323.995394,
    #             324.047498,
    #             324.099868,
    #             324.152318,
    #             324.205073,
    #             324.258542,
    #             324.313240,
    #             324.369769,
    #             324.428844,
    #             324.491353,
    #             324.558464,
    #             324.631799,
    #             324.713675,
    #         ],
    #         dtype=dtype,
    #     ),
    #     np.array(
    #         [
    #             0.002048,
    #             0.002048,
    #             0.002048,
    #             0.002048,
    #             0.002048,
    #             0.002048,
    #             0.002048,
    #             0.002048,
    #             0.002048,
    #             0.002048,
    #             0.002048,
    #             0.002048,
    #             0.002048,
    #             0.002048,
    #             0.002048,
    #             0.002048,
    #         ],
    #         dtype=dtype,
    #     ),
    #     np.empty((rows * cols), dtype=dtype),
    # )

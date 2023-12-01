# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Rodinia hotspot implementation."""

"""Computes the hotspot values for given array of tempertaure and power.

Input
---------
rows<int_64> : Number of rows for temp/power matrices.
cols<int64_t> : Number of cols for temp/power matrices.

Output
--------
result<array<float>> : Result of the hotspot computation.
Method:
The value for hotspots are determined by traversing the matrices
row-wise and computing the result for that row and then using this
as a reference for next iteration.

"""

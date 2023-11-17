# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Sycl implementation for hotspot."""
from .hotspot_sycl._hotspot_sycl import hotspot as hotspot_sycl

__all__ = ["hotspot_sycl"]

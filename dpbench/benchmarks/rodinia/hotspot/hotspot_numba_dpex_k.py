# # SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
# #
# # SPDX-License-Identifier: Apache-2.0
"""Numba-dpex implementation for hotspot."""
import dpnp
import numba_dpex

STR_SIZE = 256

# maximum power density possible (say 300W for a 10mm x 10mm chip)
MAX_PD = 3.0e6
# required precision in degrees
PRECISION = 0.001
SPEC_HEAT_SI = 1.75e6
K_SI = 100
# capacitance fitting factor
FACTOR_CHIP = 0.5
EXPAND_RATE = 2

# chip parameters
t_chip = 0.0005
chip_height = 0.016
chip_width = 0.016
BLOCK_SIZE = 16


@numba_dpex.func()
def IN_RANGE(x, min, max):
    """Check if the input value is within range.

    Args:
        x: input value.
        min: min value of range.
        max: max value of range.

    Returns: True: If value is in range.
             False: If value is out of range.
    """
    return x >= min and x <= max


@numba_dpex.func()
def MIN(a, b):
    """Return the min of two values.

    Args:
        a: first value.
        b: second value.

    Returns: a: if a <= b.
             b: if a > b.
    """
    return a if a <= b else b


@numba_dpex.kernel(debug=True)
def _hotspot_kernel(
    iteration,
    power,
    temp_src,
    temp_dst,
    grid_cols,
    grid_rows,
    border_cols,
    border_rows,
    cap,
    Rx,
    Ry,
    Rz,
    step,
):
    """Kernel to Compute the hotspot matrix for given input power and temp.

    Args:
        iteration: Number of iterations.
        power: Power matrix.
        temp_src: Src temperature array.
        temp_dst: Dest temperature array.
        grid_cols: Cols in the grid.
        grid_rows: Rows in the grid.
        border_cols: Border columns.
        border_rows: Border rows.
        cap: Temp cap.
        Rx: Rx
        Ry: Ry
        Rz: Rz
        step: Step value.

    """
    dtype = temp_src.dtype
    amb_temp = 80.0

    bx = numba_dpex.get_group_id(2)
    by = numba_dpex.get_group_id(1)

    tx = numba_dpex.get_local_id(2)
    ty = numba_dpex.get_local_id(1)

    temp_on_device = numba_dpex.private.array(
        shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=dtype
    )
    power_on_device = numba_dpex.private.array(
        shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=dtype
    )
    temp_t = numba_dpex.private.array(
        shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=dtype
    )

    step_div_cap = step / cap

    Rx_1 = 1 / Rx
    Ry_1 = 1 / Ry
    Rz_1 = 1 / Rz

    small_block_rows = BLOCK_SIZE - iteration * 2
    small_block_cols = BLOCK_SIZE - iteration * 2

    blkY = small_block_rows * by - border_rows
    blkX = small_block_cols * bx - border_cols
    blkYmax = blkY + BLOCK_SIZE - 1
    blkXmax = blkX + BLOCK_SIZE - 1

    yidx = blkY + ty
    xidx = blkX + tx

    loadYidx = yidx
    loadXidx = xidx
    index = grid_cols * loadYidx + loadXidx

    if IN_RANGE(loadYidx, 0, grid_rows - 1) and IN_RANGE(
        loadXidx, 0, grid_cols - 1
    ):
        temp_on_device[4 * 2 + 1] = temp_src[index]
        power_on_device[ty * BLOCK_SIZE + tx] = power[index]

    numba_dpex.barrier(numba_dpex.CLK_LOCAL_MEM_FENCE)

    validYmin = -blkY if blkY < 0 else 0
    validYmax = (
        BLOCK_SIZE - 1 - (blkYmax - grid_rows + 1)
        if blkYmax > grid_rows - 1
        else BLOCK_SIZE - 1
    )
    validXmin = -blkX if blkX < 0 else 0
    validXmax = (
        BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1)
        if blkXmax > grid_cols - 1
        else BLOCK_SIZE - 1
    )

    N = ty - 1
    S = ty + 1
    W = tx - 1
    E = tx + 1

    N = validYmin if N < validYmin else N
    S = validYmax if S > validYmax else S
    W = validXmin if W < validXmin else W
    E = validXmax if E > validXmax else E

    for i in range(0, iteration):
        computed = False
        if (
            IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2)
            and IN_RANGE(ty, i + 1, BLOCK_SIZE - i - 2)
            and IN_RANGE(tx, validXmin, validXmax)
            and IN_RANGE(ty, validYmin, validYmax)
        ):
            computed = True
            temp_t[ty, tx] = temp_on_device[ty, tx] + step_div_cap * (
                power_on_device[ty, tx]
                + (
                    temp_on_device[S, tx]
                    + temp_on_device[N, tx]
                    - 2.0 * temp_on_device[ty, tx]
                )
                * Ry_1
                + (
                    temp_on_device[ty, E]
                    + temp_on_device[ty, W]
                    - 2.0 * temp_on_device[ty, tx]
                )
                * Rx_1
                + (amb_temp - temp_on_device[ty, tx]) * Rz_1
            )

        numba_dpex.barrier(numba_dpex.CLK_LOCAL_MEM_FENCE)
        if i == iteration - 1:
            break
        if computed:
            temp_on_device[ty, tx] = temp_t[ty, tx]
        numba_dpex.barrier(numba_dpex.CLK_LOCAL_MEM_FENCE)

    if computed:
        temp_dst[index] = temp_t[ty, tx]


def hotspot(temperature, power, rows, cols, result):
    """Find the hotspot for given temperature and power matrix.

    Args:
        temperature: temperature matrix.
        power: power matrix.
        rows: Number of rows.
        cols: Number of cols.
        result: Result of the hotspot computation.
    """
    total_iterations = 60
    pyramid_height = 1

    size = rows * cols

    borderCols = (pyramid_height) * EXPAND_RATE / 2
    borderRows = (pyramid_height) * EXPAND_RATE / 2
    smallBlockCol = BLOCK_SIZE - (pyramid_height) * EXPAND_RATE
    smallBlockRow = BLOCK_SIZE - (pyramid_height) * EXPAND_RATE

    blockCols = int(
        cols / smallBlockCol + (0 if cols % smallBlockCol == 0 else 1)
    )
    blockRows = int(
        rows / smallBlockRow + (0 if rows % smallBlockRow == 0 else 1)
    )

    MatrixTemp = dpnp.zeros(shape=(2, size), dtype=dpnp.float32)
    MatrixTemp[0] = temperature

    dimBlock = numba_dpex.Range(1, BLOCK_SIZE, BLOCK_SIZE)
    dimGrid = numba_dpex.Range(
        1, blockRows * BLOCK_SIZE, blockCols * BLOCK_SIZE
    )

    grid_height = chip_height / rows
    grid_width = chip_width / cols

    Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height
    Rx = grid_width / (2.0 * K_SI * t_chip * grid_height)
    Ry = grid_height / (2.0 * K_SI * t_chip * grid_width)
    Rz = t_chip / (K_SI * grid_height * grid_width)

    max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI)
    step = PRECISION / max_slope
    src = 1
    dst = 0

    for t in range(0, total_iterations, pyramid_height):
        src, dst = dst, src

        _hotspot_kernel[numba_dpex.NdRange(dimGrid, dimBlock)](
            min(pyramid_height, total_iterations - t),
            power,
            MatrixTemp[src],
            MatrixTemp[dst],
            cols,
            rows,
            borderCols,
            borderRows,
            Cap,
            Rx,
            Ry,
            Rz,
            step,
        )
    k = 0
    for i in MatrixTemp[dst]:
        result[k] = i
        k += 1


temp = dpnp.array(
    [
        323.865780,
        323.898699,
        323.944688,
        323.995394,
        324.047498,
        324.099868,
        324.152318,
        324.205073,
        324.258542,
        324.313240,
        324.369769,
        324.428844,
        324.491353,
        324.558464,
        324.631799,
        324.713675,
    ],
    dtype=dpnp.float,
)
# print(type(temp))
power = dpnp.array(
    [
        0.002048,
        0.002048,
        0.002048,
        0.002048,
        0.002048,
        0.002048,
        0.002048,
        0.002048,
        0.002048,
        0.002048,
        0.002048,
        0.002048,
        0.002048,
        0.002048,
        0.002048,
        0.002048,
    ],
    dtype=dpnp.float,
)
result = dpnp.zeros((4, 4), dtype=dpnp.float)
hotspot(temp, power, 4, 4, result)
print(result)

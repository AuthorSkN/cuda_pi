# Pi by Monte-Carlo with CUDA

## Overview

Calculation Pi number with CUDA. For random values, program uses cuRand.

## How use

1. Create executable program

    ```console
    make calc_pi
    ```

2. Run

    ```console
    ./calc_pi
    ```

## System configuration

| Name  | Values  |
|-------|---------|
| CPU  | Intel® Core™ i7-8750H CPU @ 2.20GHz (Turbo Boost  4.10 GHz) × 12 |
| RAM  | 16 GB DDR4 |
| GPU  | GeForce GTX 1060 with Max-Q Design/PCIe/SSE2 |
| OS type | 64-bit  |

## Results

|   | Time, s|   PI   |
|---|--------|--------|
|CPU|0.301656|3.141348|
|GPU|0.010928|3.141348|

Number of random values for each axis is 33554432.

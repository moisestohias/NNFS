# GEMINI.MD: AI Collaboration Guide

This document provides essential context for interacting with this project. Adhering to these guidelines will ensure consistency and maintain code quality.

## 1. Project Overview & Purpose

* **Primary Goal:** This project is a neural network framework built from scratch. It provides the building blocks for creating deep learning models, including layers, activation functions, loss functions, and optimizers. **The primary goal is to provide a clear and educational implementation of a neural network library from the ground up, with a strong emphasis on performance using pure NumPy**.

## 2. Core Technologies & Stack

* **Languages:** Python version >= 3.12
* **Libraries/Dependencies:** Numpy

## 3. Architectural Patterns

* **Overall Architecture:** The project follows a modular, object-oriented architecture. Each component of a neural network (layers, activations, etc.) is encapsulated in its own class and file, promoting separation of concerns and reusability.
* **Directory Structure Philosophy:**
    * `/src`: Contains all primary source code for the neural network framework.
        * `layers.py`: Core building blocks of the network (e.g., Dense, Dropout, Conv2D..).
        * `activation.py`: Activation functions (e.g., ReLU, Softmax).
        * `functional.py`: Contains the forward and backward pass logic for the operations.
        * `tester.py`: A script to test the functionality of the framework, likely by building and training a simple model.
    * `/utils`: Contains utility functions and classes.
        * `transform.py`: Data transformation and preprocessing utilities.
        * `utils.py`: General helper functions.

## 4. Coding Conventions & Style Guide

* **Formatting:** The code generally follows PEP 8 conventions. Indentation is 4 spaces.
* **Naming Conventions:**
    * Classes are in PascalCase (e.g., `LayerDense`, `ActivationReLU`).
    * Methods and functions are in snake_case (e.g., `forward`, `backward`).
    * Variables are in snake_case.
* **Error Handling:** There is no explicit, centralized error handling strategy apparent in the provided files.
* **Implementation Philosophy & Optimization:**
    * **Efficiency is paramount.** The primary goal is to create the fastest possible pure NumPy deep learning framework. All contributions must prioritize performance.
    * **Vectorization over Loops:** Avoid explicit Python loops at all costs. All operations should be expressed as vectorized NumPy operations.
    * **`as_strided` for Convolutions and Pooling:** The `numpy.lib.stride_tricks.as_strided` function is heavily used to create efficient, loop-free implementations of convolution and pooling operations. This is a core technique of the framework and should be understood and utilized for any related contributions. The `_corr2d` function in `functional.py` is a key example of this.
    * **`im2col` / `_im_to_rows`:** The `_im_to_rows` function demonstrates the `im2col` technique, which transforms image patches into columns to allow 2D convolutions to be performed as a single matrix multiplication. This is a highly efficient approach that must be maintained.
    * **Backward Pass Implementation:** The backward pass for convolutions is implemented using transposed convolutions (sometimes referred to as "full" convolutions). The `corr2d_backward` function in `functional.py` is the reference implementation.

## 5. Key Files & Entrypoints

* **Main Entrypoint(s):** `tester.py` appears to be the main entrypoint for running a test of the framework.
* **Configuration:** There are no explicit configuration files like `.env` or `settings.py`. Configuration is likely done directly within the Python scripts.
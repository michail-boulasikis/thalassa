# pde-general

This repository contains the code generalizing the GPAI PDE solver to be able to solve PDEs with non-constant coefficients as well as non-linear PDEs.
The first paper can be found [here](https://www.overleaf.com/read/fvmxqzcfznvd#5f61e3). I will update this README when I start writing the new paper.

## Changes from PDE

There are a few changes from the original PDE solver. The main changes are:
- Use of PyTorch instead of TensorFlow. This is because PyTorch is more flexible and widely used in academia than TensorFlow. This might create some issues when running the code on Google TPUs, but I suspect that I will not get access to those again anyway.
- I will try to make use of SymPy to describe the PDEs, instead of creating my own format. This will make it easier to sell as a contribution to the field, as it will be easier to use for other researchers.
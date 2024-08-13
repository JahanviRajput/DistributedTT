# DiPTS


## Description

Method **DiPTS** (**Di**stributed **P**robabilistic **TE**nsor **S**ampling) for derivative-free optimization of the multidimensional arrays and discretized multivariate functions based on the tensor train (TT) format.

## Parameter

**Mandatory arguments**:

- `f` (function) - the target function `f(I)`, where input `I` is a 2D jax array of the shape `[samples, d]` (`d` is a number of dimensions of the function's input and `samples` is a batch size of requested multi-indices). The function should return 1D list or jax array or numpy array of the length equals to `samples` (the values of the target function for all provided multi-indices, i.e., the values of the optimized tensor).
- `d` (int) - dimension of the input tensor.
- `n` (int) - mode size for each dimension.

**Optional arguments**:

- `m` (int) - the number of allowed requests for a single black box to the objective function (the default value is `None`). If this parameter is not set, then the optimization process will continue until the objective function returns `None` instead of a list of values.
- `k` (int) - the batch size for optimization (the default value is `100`).
- `k_top` (int) - number of black boxes avaialble for parallel computation (it should be `< k`; the default value is `10`).
- `k_gd` (int) - number of gradient lifting iterations for each batch (the default value is `1`. Please note that this value ensures the maximum performance of the method, however, for a number of problems, a more accurate result is obtained by increasing this parameter, for example to `100`).
- `lr` (float): learning rate for gradient lifting iterations (the default value is `5.E-2`. Please note that this value must be correlated with the parameter `k_gd`).
- `r` (int): TT-rank of the constructed probability TT-tensor (the default value is `5`. Please note that we have not yet found problems where changing this value would lead to an improvement in the result).
- `seed` (int): parameter for jax random generator (the default value is `0`).
- `is_max` (bool): if flag is set, then maximization rather than minimization will be performed.
- `log` (bool): if flag is set, then the information about the progress of the algorithm will be printed after each improvement of the optimization result and at the end of the algorithm's work. Not that this argument may be also a function, in this case it will be used instead of ordinary `print`.
- `info` (dict): optional dictionary, which will be filled with reference information about the process of the algorithm operation.
- `P` (list): optional initial probability tensor in the TT-format (represented as a list of jax arrays, where all non-edge TT-cores are merged into one array; see the function `_generate_initial` in `protes.py` for details). If this parameter is not set, then a random initial TT-tensor will be generated. Note that this tensor will be changed inplace.



## Installation

To use this package, please install manually first the [python](https://www.python.org) programming language of the version `3.8` or `3.9`.

> To ensure version stability, we recommend working in a virtual environment, as described in the `workflow.md`. Also note that `requirements.txt` contains `jax[cpu]`; if you need the GPU version of the `jax`, please install it yourself.


## Parameters of the `dipts` function

**Mandatory arguments**:

- `f` (function) - the target function `f(I)`, where input `I` is a 2D jax array of the shape `[samples, d]` (`d` is a number of dimensions of the function's input and `samples` is a batch size of requested multi-indices). The function should return 1D list or jax array or numpy array of the length equals to `samples` (the values of the target function for all provided multi-indices, i.e., the values of the optimized tensor).
- `d` (int) - number of tensor dimensions.
- `n` (int) - mode size for each tensor's dimension.

**Optional arguments**:

- `m` (int) - the number of allowed requests to the objective function (the default value is `None`). If this parameter is not set, then the optimization process will continue until the objective function returns `None` instead of a list of values.
- `k` (int) - the batch size for optimization (the default value is `100`).
- `nbb` (int) - number of selected candidates in the batch (it should be `< k`; the default value is `10`).
- `k_gd` (int) - number of gradient lifting iterations for each batch (the default value is `1`. Please note that this value ensures the maximum performance of the method, however, for a number of problems, a more accurate result is obtained by increasing this parameter, for example to `100`).
- `lr` (float): learning rate for gradient lifting iterations (the default value is `5.E-2`. Please note that this value must be correlated with the parameter `k_gd`).
- `r` (int): TT-rank of the constructed probability TT-tensor (the default value is `5`. Please note that we have not yet found problems where changing this value would lead to an improvement in the result).
- `seed` (int): parameter for jax random generator (the default value is `0`).
- `is_max` (bool): if flag is set, then maximization rather than minimization will be performed.
- `log` (bool): if flag is set, then the information about the progress of the algorithm will be printed after each improvement of the optimization result and at the end of the algorithm's work. Not that this argument may be also a function, in this case it will be used instead of ordinary `print`.
- `info` (dict): optional dictionary, which will be filled with reference information about the process of the algorithm operation.
- `P` (list): optional initial probability tensor in the TT-format (represented as a list of jax arrays, where all non-edge TT-cores are merged into one array; see the function `_generate_initial` in `protes.py` for details). If this parameter is not set, then a random initial TT-tensor will be generated. Note that this tensor will be changed inplace.


## Notes

- You can use the outer cache for the values requested by the optimizer (that is, for each requested batch, check if any of the multi-indices have already been calculated), this can in some cases reduce the number of requests to the objective function. In this case, it is convenient not to set limits on the number of requests (parameter `m`), but instead, when the budget is exhausted (including the cache), return the `None` value from the target function (`f`), in which case the optimization algorithm will be automatically terminated.

- For a number of tasks, performance can be improved by switching to increased precision in the representation of numbers in `jax`; for this, at the beginning of the script, you should specify the code:
    ```python
    import jax
    jax.config.update('jax_enable_x64', True)
    ```

- If there is a GPU, the `jax` optimizer code will be automatically executed on it, however, the current version of the code works better on the CPU. To use CPU on a device with available GPU, you should specify the following code at the beginning of the executable script:
    ```python
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    jax.default_device(jax.devices('cpu')[0]);
    ```


## Authors

- [Jahanvi Rajput](https://github.com/JahanviRajput)
- [Drashthi Doshi](https://github.com/DrashthiD)


## Citation

If you find our approach and/or code useful in your research, please consider citing:

```bibtex
@article{Rajput2024DiPTS,
    author    = {Rajput, Jahanvi and Doshi,Drashthi},
    year      = {2024},
    title     = {{DiPTS}: {Di}stributed Probabilistic Tensor Sampling},
    journal   = {},
    url       = {}
}
```



## License

There are no restrictions on the use for scientific purposes when citing

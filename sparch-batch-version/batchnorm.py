# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    'batch_norm',
    'batch_norm_stats',
    'batch_norm_elemt',
    'batch_norm_gather_stats_with_counts',
    'batch_norm_forward',
    'batch_norm_backward_reduce',
    'batch_norm_backward_elemt',
]


def batch_norm_body(
    x, weight, bias, running_mean, running_var, total_mean, total_var,
    training: bool, eps: float, feature_axis: int = 1
):
    bound = 50.
    if training:
        mean, invstd = batch_norm_stats(x, eps, feature_axis=feature_axis)
        var = (1. / invstd) ** 2 - eps
        total_mean += mean
        total_var += (var + mean ** 2)

        run_invstd = 1. / jnp.sqrt(running_var + eps)
        condition = jnp.maximum(jnp.max(run_invstd / invstd), jnp.max(invstd / run_invstd)) <= bound
        mean = jnp.where(condition, running_mean, mean)
        invstd = jnp.where(condition, run_invstd, invstd)
    else:
        mean, invstd = running_mean, 1. / jnp.sqrt(running_var + eps)

    x = batch_norm_elemt(x, weight, bias, mean, invstd, eps, feature_axis=feature_axis)
    return x, total_mean, total_var


def batch_norm_fwd(
    x, weight, bias, running_mean, running_var, total_mean, total_var,
    training: bool, eps: float, feature_axis: int
):
    bound = 50.
    if training:
        mean, invstd = batch_norm_stats(x, eps, feature_axis=feature_axis)
        var = (1. / invstd) ** 2 - eps
        total_mean += mean
        total_var += (var + mean ** 2)

        run_invstd = 1. / jnp.sqrt(running_var + eps)
        condition = jnp.maximum(jnp.max(run_invstd / invstd), jnp.max(invstd / run_invstd)) <= bound
        mean1 = jnp.where(condition, mean, running_mean)
        invstd1 = jnp.where(condition, invstd, run_invstd)
    else:
        mean1, invstd1 = running_mean, 1. / jnp.sqrt(running_var + eps)
        invstd = invstd1
        mean = mean1

    out = batch_norm_elemt(x, weight, bias, mean1, invstd1, eps, feature_axis=feature_axis)
    return (out, total_mean, total_var), (running_mean, running_var, eps, weight, invstd, mean, x, feature_axis)


def batch_norm_bwd(training, eps, feature_axis, res, grads):
    dx, _, _ = grads
    running_mean, running_var, eps, weight, invstd, mean, x, feature_axis = res
    run_mean, run_std = running_mean, jnp.sqrt(running_var + eps)
    weight = weight / (invstd * run_std)
    sum_dy, sum_dy_xmu, grad_gamma = batch_norm_backward_reduce(
        dx, x, mean, invstd, weight, True, True, True, feature_axis=feature_axis
    )
    grad_beta = sum_dy
    grad_gamma += sum_dy * (mean - running_mean) / running_var
    grad_x = batch_norm_backward_elemt(dx, x, mean, invstd, weight, sum_dy, sum_dy_xmu, feature_axis=feature_axis)
    return grad_x, grad_gamma, grad_beta, None, None, None, None


batch_norm = jax.custom_vjp(batch_norm_body, nondiff_argnums=(7, 8, 9))
batch_norm.defvjp(batch_norm_fwd, batch_norm_bwd)


def batch_norm_stats(
    input: jnp.ndarray,
    eps: float = 1e-5,
    feature_axis: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JAX implementation of PyTorch's batch_norm_stats function.

    Computes the mean and inverse standard deviation for batch normalization.
    Statistics are computed over all dimensions except the channel dimension (axis=1).

    Args:
        input: Input tensor of shape (N, C, ...) where N is batch size, C is channels
        eps: Small value added to variance for numerical stability
        feature_axis: Axis representing the feature/channel dimension (default is 1 for NCHW format)

    Returns:
        Tuple of (mean, invstd) where:
        - mean: Mean values of shape (C,)
        - invstd: Inverse standard deviation of shape (C,), computed as 1/sqrt(var + eps)
    """

    # Determine axes to reduce over (all except feature_axis)
    ndim = input.ndim
    feature_axis = feature_axis + ndim if feature_axis < 0 else feature_axis
    reduce_axes = tuple(i for i in range(ndim) if i != feature_axis)

    # Compute mean along the specified axes
    mean = jnp.mean(input, axis=reduce_axes, keepdims=False)

    # Compute variance along the specified axes
    var = jnp.var(input, axis=reduce_axes, keepdims=False)

    # Compute inverse standard deviation
    invstd = 1.0 / jnp.sqrt(var + eps)
    return mean, invstd


def batch_norm_elemt(
    input: jnp.ndarray,
    weight: Optional[jnp.ndarray],
    bias: Optional[jnp.ndarray],
    mean: jnp.ndarray,
    invstd: jnp.ndarray,
    eps: float = 1e-5,
    feature_axis: int = 1,
) -> jnp.ndarray:
    """
    JAX implementation of PyTorch's batch_norm_elemt function.

    Applies batch normalization element-wise using pre-computed statistics.

    Args:
        input: Input tensor of shape (N, C, ...)
        weight: Scale parameter (gamma) of shape (C,), can be None
        bias: Shift parameter (beta) of shape (C,), can be None
        mean: Mean values of shape (C,)
        invstd: Inverse standard deviation of shape (C,)
        eps: Small value for numerical stability (not used here since invstd is pre-computed)
        feature_axis: Axis representing the feature/channel dimension (default is 1 for NCHW format)

    Returns:
        Normalized tensor of the same shape as input
    """

    # Determine axes to expand for broadcasting
    ndim = input.ndim
    feature_axis = feature_axis + ndim if feature_axis < 0 else feature_axis
    expand_axes = tuple(i for i in range(ndim) if i != feature_axis)

    # Expand mean and invstd for broadcasting
    if expand_axes:
        expanded_mean = jnp.expand_dims(mean, axis=expand_axes)
        expanded_invstd = jnp.expand_dims(invstd, axis=expand_axes)
    else:
        expanded_mean = mean
        expanded_invstd = invstd

    # Apply normalization: (input - mean) * invstd
    normalized = (input - expanded_mean) * expanded_invstd

    # Apply weight (gamma) if provided
    if weight is not None:
        if expand_axes:
            expanded_weight = jnp.expand_dims(weight, axis=expand_axes)
        else:
            expanded_weight = weight
        normalized = normalized * expanded_weight

    # Apply bias (beta) if provided
    if bias is not None:
        if expand_axes:
            expanded_bias = jnp.expand_dims(bias, axis=expand_axes)
        else:
            expanded_bias = bias
        normalized = normalized + expanded_bias

    return normalized


def batch_norm_gather_stats_with_counts(
    mean: jnp.ndarray,
    invstd: jnp.ndarray,
    running_mean: Optional[jnp.ndarray] = None,
    running_var: Optional[jnp.ndarray] = None,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Update running statistics using current batch statistics.
    This is similar to PyTorch's batch_norm_gather_stats_with_counts.

    Args:
        mean: Current batch mean
        invstd: Current batch inverse standard deviation
        running_mean: Running mean to update
        running_var: Running variance to update
        momentum: Momentum for the exponential moving average
        eps: Small value for numerical stability
        count: Number of elements in the batch
        feature_axis: Axis representing the feature/channel dimension (default is 1 for NCHW format)

    Returns:
        Tuple of (updated_running_mean, updated_running_var)
    """

    if running_mean is None:
        running_mean = jnp.zeros_like(mean)
    if running_var is None:
        running_var = jnp.ones_like(mean)

    # Convert invstd back to variance for running statistics
    current_var = 1.0 / (invstd ** 2) - eps

    # Update running statistics using exponential moving average
    updated_running_mean = (1 - momentum) * running_mean + momentum * mean
    updated_running_var = (1 - momentum) * running_var + momentum * current_var

    return updated_running_mean, updated_running_var


def batch_norm_forward(
    input: jnp.ndarray,
    weight: Optional[jnp.ndarray] = None,
    bias: Optional[jnp.ndarray] = None,
    running_mean: Optional[jnp.ndarray] = None,
    running_var: Optional[jnp.ndarray] = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Complete batch normalization forward pass combining stats and elemt functions.

    Args:
        input: Input tensor
        weight: Scale parameter (gamma)
        bias: Shift parameter (beta)
        running_mean: Running mean for inference
        running_var: Running variance for inference
        training: Whether in training mode
        momentum: Momentum for running stats update
        eps: Small value for numerical stability

    Returns:
        Tuple of (output, mean, invstd, updated_running_mean, updated_running_var)
    """

    if training:
        # Compute batch statistics
        mean, invstd = batch_norm_stats(input, eps)

        # Apply normalization
        output = batch_norm_elemt(input, weight, bias, mean, invstd, eps)

        # Update running statistics
        updated_running_mean, updated_running_var = batch_norm_gather_stats_with_counts(
            mean, invstd, running_mean, running_var, momentum, eps
        )

    else:
        # Use running statistics for inference
        if running_mean is None or running_var is None:
            raise ValueError("Running statistics must be provided for inference mode")

        mean = running_mean
        invstd = 1.0 / jnp.sqrt(running_var + eps)

        # Apply normalization
        output = batch_norm_elemt(input, weight, bias, mean, invstd, eps)

        updated_running_mean = running_mean
        updated_running_var = running_var

    return output, mean, invstd, updated_running_mean, updated_running_var


def batch_norm_backward_reduce(
    grad_output: jnp.ndarray,
    input: jnp.ndarray,
    mean: jnp.ndarray,
    invstd: jnp.ndarray,
    weight: Optional[jnp.ndarray] = None,
    input_g: bool = True,
    weight_g: bool = True,
    bias_g: bool = True,
    feature_axis: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JAX implementation of PyTorch's batch_norm_backward_reduce function.

    This function computes the gradients for gamma (weight) and beta (bias) parameters
    in batch normalization backward pass, as well as intermediate values for input gradient.

    Args:
        grad_output: Gradient of the loss w.r.t. the output of batch norm
        input: Input tensor to batch normalization
        mean: Mean values computed during forward pass
        invstd: Inverse standard deviation computed during forward pass (1/sqrt(var + eps))
        weight: Scale parameter (gamma), optional
        input_g: Whether to compute input gradient components
        weight_g: Whether to compute weight gradient
        bias_g: Whether to compute bias gradient
        feature_axis: Axis representing the feature/channel dimension (default is 1 for NCHW format)

    Returns:
        Tuple of (sum_dy, sum_dy_xmu, grad_weight) where:
        - sum_dy: Sum of grad_output, used for bias gradient
        - sum_dy_xmu: Sum of grad_output * normalized_input, used for input gradient
        - grad_weight: Gradient w.r.t. weight (gamma)
    """

    # Determine axes to reduce over (all except feature_axis)
    ndim = input.ndim
    reduce_axes = tuple(i for i in range(ndim) if i != feature_axis)

    # Compute sum of gradients (for bias gradient)
    if bias_g:
        sum_dy = jnp.sum(grad_output, axis=reduce_axes, keepdims=False)
    else:
        sum_dy = jnp.zeros_like(mean)

    # Compute normalized input: (input - mean) * invstd
    if input_g or weight_g:
        # Expand mean and invstd to match input shape for broadcasting
        expanded_mean = jnp.expand_dims(mean, axis=reduce_axes)
        expanded_invstd = jnp.expand_dims(invstd, axis=reduce_axes)

        # Normalize input
        normalized_input = (input - expanded_mean) * expanded_invstd

        # Compute sum of grad_output * normalized_input (for input gradient computation)
        if input_g:
            sum_dy_xmu = jnp.sum(grad_output * normalized_input, axis=reduce_axes, keepdims=False)
        else:
            sum_dy_xmu = jnp.zeros_like(mean)

        # Compute weight gradient: sum(grad_output * normalized_input)
        if weight_g:
            grad_weight = sum_dy_xmu.copy() if weight is not None else sum_dy_xmu
        else:
            grad_weight = jnp.zeros_like(mean) if weight is None else jnp.zeros_like(weight)
    else:
        sum_dy_xmu = jnp.zeros_like(mean)
        grad_weight = jnp.zeros_like(mean) if weight is None else jnp.zeros_like(weight)

    return sum_dy, sum_dy_xmu, grad_weight


def batch_norm_backward_elemt(
    grad_output: jnp.ndarray,
    input: jnp.ndarray,
    mean: jnp.ndarray,
    invstd: jnp.ndarray,
    weight: Optional[jnp.ndarray] = None,
    sum_dy: Optional[jnp.ndarray] = None,
    sum_dy_xmu: Optional[jnp.ndarray] = None,
    count: Optional[int] = None,
    feature_axis: int = 1,
) -> jnp.ndarray:
    """
    Compute the input gradient for batch normalization.
    This is typically called after batch_norm_backward_reduce.

    Args:
        grad_output: Gradient of the loss w.r.t. the output
        input: Input tensor
        mean: Mean values from forward pass
        invstd: Inverse standard deviation from forward pass
        weight: Scale parameter (gamma)
        sum_dy: Sum of grad_output (from reduce function)
        sum_dy_xmu: Sum of grad_output * normalized_input (from reduce function)
        count: Number of elements in the batch (for normalization)
        feature_axis: Axis representing the feature/channel dimension (default is 1 for NCHW format)

    Returns:
        Gradient w.r.t. input
    """

    # Determine axes to reduce over (all except feature_axis)
    ndim = input.ndim
    reduce_axes = tuple(i for i in range(ndim) if i != feature_axis)
    count = np.prod(np.array([input.shape[i] for i in reduce_axes])) if count is None else count

    # Expand tensors for broadcasting
    expanded_mean = jnp.expand_dims(mean, axis=reduce_axes)
    expanded_invstd = jnp.expand_dims(invstd, axis=reduce_axes)
    expanded_weight = jnp.expand_dims(weight, axis=reduce_axes) if weight is not None else 1.0

    if sum_dy is not None and sum_dy_xmu is not None:
        expanded_sum_dy = jnp.expand_dims(sum_dy, axis=reduce_axes)
        expanded_sum_dy_xmu = jnp.expand_dims(sum_dy_xmu, axis=reduce_axes)
    else:
        expanded_sum_dy = jnp.sum(grad_output, axis=reduce_axes, keepdims=True)
        normalized_input = (input - expanded_mean) * expanded_invstd
        expanded_sum_dy_xmu = jnp.sum(grad_output * normalized_input, axis=reduce_axes, keepdims=True)

    # Compute input gradient
    normalized_input = (input - expanded_mean) * expanded_invstd

    grad_input = expanded_weight * expanded_invstd * (
        grad_output - expanded_sum_dy / count -
        normalized_input * expanded_sum_dy_xmu / count
    )
    return grad_input


def try1():
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)

    print("=== Testing batch_norm_stats ===")

    # Test with 4D tensor (NCHW format)
    input_4d = jax.random.normal(key, (2, 3, 4, 4))
    print(f"Input 4D shape: {input_4d.shape}")

    mean, invstd = batch_norm_stats(input_4d, eps=1e-5)
    print(f"Mean shape: {mean.shape}, values: {mean}")
    print(f"Invstd shape: {invstd.shape}, values: {invstd}")

    # Test with 2D tensor (NC format)
    input_2d = jax.random.normal(jax.random.split(key)[0], (10, 5))
    print(f"\nInput 2D shape: {input_2d.shape}")

    mean_2d, invstd_2d = batch_norm_stats(input_2d, eps=1e-5)
    print(f"Mean 2D shape: {mean_2d.shape}, values: {mean_2d}")
    print(f"Invstd 2D shape: {invstd_2d.shape}, values: {invstd_2d}")

    print("\n=== Testing batch_norm_elemt ===")

    # Create weight and bias parameters
    num_channels = 3
    weight = jax.random.normal(jax.random.split(key, 2)[0], (num_channels,))
    bias = jax.random.normal(jax.random.split(key, 2)[1], (num_channels,))

    # Apply batch normalization
    output = batch_norm_elemt(input_4d, weight, bias, mean, invstd)
    print(f"Output shape: {output.shape}")
    print(f"Output mean per channel: {jnp.mean(output, axis=(0, 2, 3))}")
    print(f"Output std per channel: {jnp.std(output, axis=(0, 2, 3))}")

    print("\n=== Testing complete forward pass ===")

    # Test complete batch norm forward pass
    running_mean = jnp.zeros(num_channels)
    running_var = jnp.ones(num_channels)

    output, batch_mean, batch_invstd, new_running_mean, new_running_var = batch_norm_forward(
        input_4d, weight, bias, running_mean, running_var, training=True
    )

    print(f"Training mode output shape: {output.shape}")
    print(f"Updated running mean: {new_running_mean}")
    print(f"Updated running var: {new_running_var}")

    # Test inference mode
    output_inf, _, _, _, _ = batch_norm_forward(
        input_4d, weight, bias, new_running_mean, new_running_var, training=False
    )

    print(f"Inference mode output shape: {output_inf.shape}")
    print("Batch norm implementation complete!")


def try2():
    # Example with 4D tensor (batch_size=2, channels=3, height=4, width=4)
    key = jax.random.PRNGKey(42)

    # Create sample data
    input_tensor = jax.random.normal(key, (2, 3, 4, 4))
    grad_output = jax.random.normal(jax.random.split(key)[0], (2, 3, 4, 4))

    # Simulate forward pass statistics
    mean = jnp.mean(input_tensor, axis=(0, 2, 3))
    var = jnp.var(input_tensor, axis=(0, 2, 3))
    eps = 1e-5
    invstd = 1.0 / jnp.sqrt(var + eps)
    weight = jnp.ones(3)  # gamma parameter

    # Compute backward reduce
    sum_dy, sum_dy_xmu, grad_weight = batch_norm_backward_reduce(
        grad_output, input_tensor, mean, invstd, weight
    )

    # Compute input gradient
    grad_input = batch_norm_backward_elemt(
        grad_output, input_tensor, mean, invstd, weight, sum_dy, sum_dy_xmu
    )

    print("Sum of grad_output (bias gradient):", sum_dy)
    print("Sum of grad_output * normalized_input:", sum_dy_xmu)
    print("Weight gradient:", grad_weight)
    print("Input gradient shape:", grad_input.shape)


def try3():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, 3, 8, 8))
    weight = jnp.ones(3)
    bias = jnp.zeros(3)
    running_mean = jnp.zeros(3)
    running_var = jnp.ones(3)
    total_mean = jnp.zeros(3)
    total_var = jnp.zeros(3)
    eps = 1e-5

    # Training mode
    out, tmean, tvar = batch_norm(
        x, weight, bias, running_mean, running_var, total_mean, total_var,
        training=True, eps=eps, feature_axis=1
    )
    print("Training output shape:", out.shape)
    print("Total mean:", tmean)
    print("Total var:", tvar)

    def f(weight):
        out, tmean, tvar = batch_norm(
            x, weight, bias, running_mean, running_var, total_mean, total_var,
            training=True, eps=eps, feature_axis=1
        )
        return out.sum()

    dg = jax.grad(f)(weight)


if __name__ == '__main__':
    # try1()
    # try2()
    try3()

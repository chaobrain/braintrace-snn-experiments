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

# -*- coding: utf-8 -*-

import functools
from typing import Callable

import jax
import jax.numpy as jnp

import brainscale
import brainstate

__all__ = [
    'get_neuron',
    'FFNetSHD',
    'RecNetSHD',
]


class Exponential(brainstate.surrogate.Surrogate):
    def surrogate_grad(self, x) -> jax.Array:
        return jnp.exp(-jnp.abs(x))


class Triangle(brainstate.surrogate.Surrogate):
    """Altered from code of Temporal Efficient Training, ICLR 2022 (https://openreview.net/forum?id=_XNtisL32jv)
    max(0, 1 − |ui[t] − θ|)"""

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def surrogate_grad(self, x) -> jax.Array:
        return jnp.maximum(self.gamma - jnp.abs(x), 0.) / self.gamma ** 2


class Rectangle(brainstate.surrogate.Surrogate):
    def surrogate_grad(self, x) -> jax.Array:
        return jnp.asarray(jnp.abs(x) < 0.5, dtype=x.dtype)


def get_surrogate(name: str) -> brainstate.surrogate.Surrogate:
    if name == 'exp':
        return Exponential()
    elif name == 'triangle':
        return Triangle()
    elif name == 'rectangle':
        return Rectangle()
    else:
        raise NotImplementedError


class LIF(brainstate.nn.Neuron):
    """
    Leaky Integrate-and-Fire (LIF) neuron model.

    This class implements a LIF neuron model with configurable parameters for decay,
    threshold, spike function, and reset behavior.

    Parameters:
    -----------
    in_size : brainstate.typing.Size
        The input size of the neuron.
    decay_factor : float, optional
        The decay factor for the membrane potential (default is 1.0).
    v_threshold : float, optional
        The threshold voltage for spike generation (default is 1.0).
    spk_fun : Callable, optional
        The spike function to use (default is brainstate.surrogate.ReluGrad()).
    hard_reset : bool, optional
        If True, performs a hard reset of membrane potential after spike (default is False).
    detach_reset : bool, optional
        If True, detaches the gradient for the reset operation (default is False).
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        decay_factor: float = 1.,
        v_threshold: float = 1.,
        spk_fun: Callable = brainstate.surrogate.ReluGrad(),
        hard_reset: bool = False,
        detach_reset: bool = False
    ):
        super().__init__(in_size)

        self.decay_factor = decay_factor
        self.v_threshold = v_threshold
        self.hard_reset = hard_reset
        self.detach_reset = detach_reset
        assert callable(spk_fun), 'spk_fun must be callable'
        self.spk_fun = spk_fun

    def init_state(self, *args, **kwargs):
        """
        Initialize the neuron state.

        This method initializes the membrane potential (v) of the neuron.
        """
        self.v = brainscale.ETraceState(brainstate.init.param(jnp.zeros, self.varshape))

    def update(self, x):
        """
        Update the neuron state and generate spikes.

        This method updates the membrane potential based on the input and current state,
        generates spikes, and applies the reset mechanism.

        Parameters:
        -----------
        x : array_like
            The input to the neuron.

        Returns:
        --------
        spk : array_like
            The generated spikes.
        """
        v = self.v.value

        # neuronal reset
        spk = self.get_spike(v)
        if self.detach_reset:
            spike_d = jax.lax.stop_gradient(spk)
        else:
            spike_d = spk
        if self.hard_reset:
            v = v * (1. - spike_d)
        else:
            v = v - spike_d * self.v_threshold

        # neuronal charge
        v = v * self.decay_factor + x
        self.v.value = v

        # neuronal spike
        return self.get_spike(v)

    def get_spike(self, v=None):
        if v is None:
            v = self.v.value
        return self.spk_fun(v - self.v_threshold)


class KLIF(brainstate.nn.Neuron):
    def __init__(
        self,
        in_size: brainstate.typing.Size,
        tau: float = 2.,
        decay_input: bool = True,
        v_threshold: float = 1.,
        v_reset: float = 0.,
        spk_fun: Callable = brainstate.surrogate.ReluGrad(),
        detach_reset: bool = False,
        hard_reset: bool = False,
    ):
        super().__init__(in_size)

        assert isinstance(tau, float) and tau > 1.

        self.tau = tau
        self.decay_input = decay_input
        ones = jnp.ones(self.varshape)
        self.k = brainscale.ElemWiseParam(1., lambda w: w * ones)

        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.hard_reset = hard_reset
        assert callable(spk_fun), 'spk_fun must be callable'
        self.spk_fun = spk_fun

    def init_state(self, *args, **kwargs):
        """
        Initialize the neuron state.

        This method initializes the membrane potential (v) of the neuron.
        """
        self.v = brainscale.ETraceState(brainstate.init.param(jnp.zeros, self.varshape))

    def update(self, x):
        v = self.v.value
        k = self.k.execute()

        # neuronal reset
        spk = self.get_spike(v)
        if self.detach_reset:
            spike_d = jax.lax.stop_gradient(spk)
        else:
            spike_d = spk
        if self.hard_reset:
            v = (1. - spike_d) * v + spike_d * self.v_reset
        else:
            v = v - spike_d * self.v_threshold

        # neuronal charge
        if self.decay_input:
            v = v + (x - (v - self.v_reset)) / self.tau
            v = jax.nn.relu(k * v)
        else:
            v = v - (v - self.v_reset) / self.tau + x
            v = jax.nn.relu(k * v)
        self.v.value = v

        # neuronal fire
        return self.get_spike(v)

    def get_spike(self, v=None):
        if v is None:
            v = self.v.value
        return self.spk_fun(v - self.v_threshold)


class ParametricLIF(brainstate.nn.Neuron):
    def __init__(
        self,
        in_size: brainstate.typing.Size,
        tau: Callable = brainstate.init.Constant(-1.4),
        decay_input: bool = False,
        v_threshold: float = 1.,
        v_reset: float = 0.,
        spk_fun: Callable = brainstate.surrogate.ReluGrad(),
        detach_reset: bool = False,
        hard_reset: bool = False,
    ):
        super().__init__(in_size)

        self.w = brainscale.ElemWiseParam(tau(self.varshape))

        self.decay_input = decay_input
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.hard_reset = hard_reset
        assert callable(spk_fun), 'spk_fun must be callable'
        self.spk_fun = spk_fun

    def init_state(self, *args, **kwargs):
        """
        Initialize the neuron state.

        This method initializes the membrane potential (v) of the neuron.
        """
        self.v = brainscale.ETraceState(brainstate.init.param(jnp.zeros, self.varshape))

    def update(self, x):
        v = self.v.value
        w = jax.nn.sigmoid(self.w.execute())

        # neuronal reset
        spk = self.spk_fun(v - self.v_threshold)
        if self.detach_reset:
            spike_d = jax.lax.stop_gradient(spk)
        else:
            spike_d = spk
        if self.hard_reset:
            v = (1. - spike_d) * v + spike_d * self.v_reset
        else:
            v = v - spike_d * self.v_threshold

        # neuronal charge
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                v = v + (x - v) * w
            else:
                v = v + (x - (v - self.v_reset)) * w
        else:
            if self.v_reset is None or self.v_reset == 0.:
                v = v * (1. - w) + x
            else:
                v = v - (v - self.v_reset) * w + x
        self.v.value = v

        # neuronal fire
        return self.get_spike(v)

    def get_spike(self, v=None):
        if v is None:
            v = self.v.value
        return self.spk_fun(v - self.v_threshold)


class TwoCompartmentLIF(brainstate.nn.Dynamics):
    r"""
    A Two-Compartment Spiking Neuron Model for Long-Term Sequential Modelling.

    $$
    \begin{aligned}
    U_D[t] &= U_D[t - 1] + \beta_1 U_S[t - 1] + I[t] - \gamma S[t - 1], \\
    U_S[t] &= U_S[t - 1] + \beta_2 U_D[t] - V_{th} S[t - 1], \\
    S[t] &= \Theta(U_S[t] - V_{th}),
    \end{aligned}
    $$

    where $\mathcal{U}^D$ and $\mathcal{U}^S$ represents the membrane
    potentials of the dendritic and the somatic compartments, respectively.
    $\alpha_1$ and $\alpha_2$ are respective membrane potential decaying
    coefficients for these two compartments (have been dropped).
    Notably, the membrane potentials
    of these two compartments are not updated independently. Rather, they
    are coupled with each other through the second term in above equations,
    in which the coupling effects are controlled by the coefficients
    $\beta_1$ and $\beta_2$. The interplay between these two compartments
    enhances the neuronal dynamics and, if properly designed, can resolve
    the vanishing gradient problem.

    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        v_threshold: float = 1.,
        v_reset: float = 0.,
        detach_reset: bool = False,
        hard_reset: bool = False,
        beta1: float = 0.,
        beta2: float = 0.,
        gamma: float = 0.5,
        spk_fun: Callable = brainstate.surrogate.ReluGrad(),
    ):
        super().__init__(in_size)

        ones = jnp.ones(self.varshape, dtype=brainstate.environ.dftype())
        # self.beta1 = brainscale.ElemWiseParam(beta1, lambda w: w * ones)
        # self.beta2 = brainscale.ElemWiseParam(beta2, lambda w: w * ones)
        self.beta1 = brainscale.ElemWiseParam(jnp.ones(self.varshape) * beta1)
        self.beta2 = brainscale.ElemWiseParam(jnp.ones(self.varshape) * beta2)

        self.gamma = gamma
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.hard_reset = hard_reset
        assert callable(spk_fun), 'spk_fun must be callable'
        self.spk_fun = spk_fun

    def init_state(self, *args, **kwargs):
        """
        Initialize the neuron state.

        This method initializes the membrane potential (v) of the neuron.
        """
        self.vd = brainscale.ETraceState(brainstate.init.param(jnp.zeros, self.varshape))
        self.vs = brainscale.ETraceState(brainstate.init.param(jnp.zeros, self.varshape))

    def update(self, x):
        vd = self.vd.value
        vs = self.vs.value

        # neuronal reset
        spk = self.get_spike(vs)
        if self.detach_reset:
            spike_d = jax.lax.stop_gradient(spk)
        else:
            spike_d = spk
        if not self.hard_reset:
            jit_soft_reset = lambda v, sp, v_threshold: v - sp * v_threshold
            vd = jit_soft_reset(vd, spike_d, self.gamma)
            vs = jit_soft_reset(vs, spike_d, self.v_threshold)
        else:
            hard_reset = lambda v, sp, v_reset: (1. - sp) * v + sp * v_reset
            vs = hard_reset(vs, spike_d, self.v_reset)

        # neuronal charge
        beta1 = jax.nn.sigmoid(self.beta1.execute())
        beta2 = jax.nn.sigmoid(self.beta2.execute())
        vd = vd - beta1 * vs + x
        vs = vs + beta2 * vd

        # neuronal_fire
        self.vd.value = vd
        self.vs.value = vs
        return self.get_spike(vs)

    def get_spike(self, v=None):
        if v is None:
            v = self.vs.value
        return self.spk_fun(v - self.v_threshold)


def get_neuron(args):
    # surrogate function
    surrogate = get_surrogate(args.sg)

    if args.neuron == 'plif':
        node = functools.partial(
            ParametricLIF,
            v_threshold=args.threshold,
            spk_fun=surrogate,
            hard_reset=args.hard_reset,
            detach_reset=False,
        )
    elif args.neuron == 'lif':
        node = functools.partial(
            LIF,
            v_threshold=args.threshold,
            spk_fun=surrogate,
            hard_reset=args.hard_reset,
            detach_reset=False,
        )
    elif args.neuron == 'tclif':
        node = functools.partial(
            TwoCompartmentLIF,
            v_threshold=args.threshold,
            spk_fun=surrogate,
            hard_reset=args.hard_reset,
            detach_reset=False,
            gamma=args.gamma,
            beta1=args.beta1,
            beta2=args.beta2
        )
        print(f"beta init from {jax.nn.sigmoid(args.beta1):.2f} and {jax.nn.sigmoid(args.beta2):.2f}")
    else:
        raise NotImplementedError
    return node


class FFNetSHD(brainstate.nn.Module):
    """
    Feed-Forward Neural Network for Sequential Hierarchical Decision-making (FFNetSHD).

    This class implements a feed-forward neural network architecture designed for
    sequential hierarchical decision-making tasks. It consists of multiple layers
    including linear transformations, dropout, and spiking neurons.

    Parameters:
    -----------
    in_dim : int, optional
        The input dimension of the network (default is 8).
    hidden : int, optional
        The number of hidden units in each layer (default is 128).
    out_dim : int, optional
        The output dimension of the network (default is 20).
    spiking_neuron : callable, optional
        A function that returns a spiking neuron instance (default is None).
    drop : float, optional
        The dropout probability (default is 0.0).

    Attributes:
    -----------
    features : brainstate.nn.Sequential
        A sequential container of the network layers.
    """

    def __init__(self, in_dim=8, hidden=128, out_dim=20, spiking_neuron=None, drop=0.0):
        super().__init__()
        layers = []
        layers += [
            brainscale.nn.Linear(in_dim, hidden),
            brainscale.nn.Dropout(1 - drop),
            spiking_neuron(hidden)
        ]
        layers += [
            brainscale.nn.Linear(hidden, hidden),
            brainscale.nn.Dropout(1 - drop),
            spiking_neuron(hidden)
        ]
        layers += [brainscale.nn.Linear(hidden, out_dim)]
        self.features = brainstate.nn.Sequential(*layers)

    def update(self, x):
        """
        Forward pass of the FFNetSHD.

        This method processes the input through the network layers and returns the output.

        Parameters:
        -----------
        x : array_like
            The input data. Must be a 1-dimensional array with shape [n_features].

        Returns:
        --------
        array_like
            The output of the network after processing the input.

        Raises:
        -------
        AssertionError
            If the input `x` is not a 1-dimensional array.
        """
        assert x.ndim == 1, 'require data with the shape of [n_feature]'
        return self.features(x)


class RecurrentContainer(brainstate.nn.Module):
    def __init__(self, neuron: brainstate.nn.Neuron, ):
        super().__init__()

        self.neuron = neuron
        self.recurrent_weight = brainscale.nn.Linear(neuron.in_size, neuron.in_size)

    def update(self, ff_x):
        rec_x = self.recurrent_weight(self.neuron.get_spike())
        return self.neuron(rec_x + ff_x)


class RecNetSHD(brainstate.nn.Module):
    """
    Recurrent Neural Network for Sequential Hierarchical Decision-making (RecNetSHD).

    This class implements a recurrent neural network architecture designed for
    sequential hierarchical decision-making tasks. It consists of multiple layers
    including linear transformations, dropout, and recurrent spiking neuron containers.

    Parameters:
    -----------
    in_dim : int, optional
        The input dimension of the network (default is 8).
    hidden : int, optional
        The number of hidden units in each layer (default is 128).
    out_dim : int, optional
        The output dimension of the network (default is 20).
    spiking_neuron : callable, optional
        A function that returns a spiking neuron instance (default is None).
    drop : float, optional
        The dropout probability (default is 0.0).

    Attributes:
    -----------
    features : brainstate.nn.Sequential
        A sequential container of the network layers.
    """

    def __init__(self, in_dim=8, hidden=128, out_dim=20, spiking_neuron=None, drop=0.0):
        super().__init__()
        layers = []
        layers += [
            brainscale.nn.Linear(in_dim, hidden),
            brainscale.nn.Dropout(1. - drop),
            RecurrentContainer(spiking_neuron(hidden))
        ]
        layers += [
            brainscale.nn.Linear(hidden, hidden),
            brainscale.nn.Dropout(1. - drop),
            RecurrentContainer(spiking_neuron(hidden))
        ]
        layers += [brainscale.nn.Linear(hidden, out_dim)]
        self.features = brainstate.nn.Sequential(*layers)

    def update(self, x):
        """
        Update the network state and compute the output for a given input.

        This method processes the input through the network layers and returns the output.

        Parameters:
        -----------
        x : array_like
            The input data. Must be a 1-dimensional array with shape [n_features].

        Returns:
        --------
        array_like
            The output of the network after processing the input.

        Raises:
        -------
        AssertionError
            If the input `x` is not a 1-dimensional array.
        """
        assert x.ndim == 1, 'require data with the shape of [n_feature]'
        return self.features(x)

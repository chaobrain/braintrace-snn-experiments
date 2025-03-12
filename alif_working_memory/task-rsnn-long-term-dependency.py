# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

import argparse
import os.path
import pickle
import platform
import time
from functools import reduce
from typing import Any, Callable, Union

import numpy as np

from utils import MyArgumentParser

parser = MyArgumentParser()

# Learning parameters
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs.")
parser.add_argument("--dt", type=float, default=1., help="The simulation time step.")

# dataset
parser.add_argument("--mode", type=str, default="train", choices=['train', 'sim'], help="The computing mode.")
parser.add_argument("--n_data_worker", type=int, default=1, help="Number of data loading workers (default: 4)")
parser.add_argument("--t_delay", type=float, default=1e3, help="Deta delay length.")
parser.add_argument("--t_fixation", type=float, default=500., help="")

# training parameters
parser.add_argument("--exp_name", type=str, default='', help="")
parser.add_argument("--warmup_ratio", type=float, default=0.0, help="The ratio for network simulation.")
parser.add_argument("--acc_th", type=float, default=0.95, help="")
parser.add_argument("--filepath", type=str, default='', help="The name for the current experiment.")

# regularization parameters
parser.add_argument("--spk_reg_factor", type=float, default=0.0, help="Spike regularization factor.")
parser.add_argument("--spk_reg_rate", type=float, default=10., help="Target firing rate.")
parser.add_argument("--v_reg_factor", type=float, default=0.0, help="Voltage regularization factor.")
parser.add_argument("--v_reg_low", type=float, default=-20., help="The lowest voltage for regularization.")
parser.add_argument("--v_reg_high", type=float, default=1.4, help="The highest voltage for regularization.")
parser.add_argument("--weight_L1", type=float, default=0.0, help="The weight L1 regularization.")
parser.add_argument("--weight_L2", type=float, default=0.0, help="The weight L2 regularization.")

# GIF parameters
parser.add_argument("--diff_spike", type=int, default=0, help="0: False, 1: True.")
parser.add_argument("--n_rec", type=int, default=200, help="Number of recurrent neurons.")
parser.add_argument("--A2", type=float, default=-1.)
parser.add_argument("--tau_I2", type=float, default=2000.)
parser.add_argument("--tau_neu", type=float, default=20.)
parser.add_argument("--tau_syn", type=float, default=10.)
parser.add_argument("--tau_o", type=float, default=10.)
parser.add_argument("--ff_scale", type=float, default=10.)
parser.add_argument("--rec_scale", type=float, default=2.)

global_args = parser.parse_args()

import matplotlib

if platform.platform().startswith('Linux'):
    matplotlib.use('agg')

import matplotlib.pyplot as plt
import brainscale
import brainstate
import braintools
import brainunit as u
import brainpy_datasets as bd
import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader, IterableDataset
from numba import njit

PyTree = Any

diag_norm_mapping = {
    0: None,
    1: True,
    2: False
}


def format_sim_epoch(sim: Union[int, float], length: int):
    if 0. <= sim < 1.:
        return int(length * sim)
    else:
        return int(sim)


def raster_plot(sp_matrix, times):
    """Get spike raster plot which displays the spiking activity
    of a group of neurons over time.
  
    Parameters
    ----------
    sp_matrix : bnp.ndarray
        The matrix which record spiking activities.
    times : bnp.ndarray
        The time steps.
  
    Returns
    -------
    raster_plot : tuple
        Include (neuron index, spike time).
    """
    sp_matrix = np.asarray(sp_matrix)
    times = np.asarray(times)
    elements = np.where(sp_matrix > 0.)
    index = elements[1]
    times = times[elements[0]]
    return index, times


class ExponentialSmooth(object):
    def __init__(self, decay: float = 0.8):
        self.decay = decay
        self.value = None

    def update(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * value
        return self.value

    def __call__(self, value, i: int = None):
        return self.update(value)  # / (1. - self.decay ** (i + 1))


class GIF(brainstate.nn.Neuron):
    def __init__(
        self, size,
        V_rest=0., V_th_inf=1., R=1., tau=20., tau_I2=50., A2=0.,
        V_initializer: Callable = brainstate.init.Constant(1.),
        I2_initializer: Callable = brainstate.init.ZeroInit(),
        spike_fun: Callable = brainstate.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        keep_size: bool = False,
        name: str = None,
        mode: brainstate.mixin.Mode = None,
    ):
        super().__init__(size, name=name, spk_fun=spike_fun, spk_reset=spk_reset)

        # params
        self.V_rest = brainstate.init.param(V_rest, self.varshape, allow_none=False)
        self.V_th_inf = brainstate.init.param(V_th_inf, self.varshape, allow_none=False)
        self.R = brainstate.init.param(R, self.varshape, allow_none=False)
        self.tau = brainstate.init.param(tau, self.varshape, allow_none=False)
        self.tau_I2 = brainstate.init.param(tau_I2, self.varshape, allow_none=False)
        self.A2 = brainstate.init.param(A2, self.varshape, allow_none=False)

        # initializers
        self._V_initializer = V_initializer
        self._I2_initializer = I2_initializer

    def init_state(self, batch_size=None):
        self.V = brainscale.ETraceState(brainstate.init.param(self._V_initializer, self.varshape, batch_size))
        self.I2 = brainscale.ETraceState(brainstate.init.param(self._I2_initializer, self.varshape, batch_size))

    def dI2(self, I2, t):
        return - I2 / self.tau_I2

    def dV(self, V, t, I_ext):
        return (- V + self.V_rest + self.R * I_ext) / self.tau

    def update(self, x=0.):
        t = brainstate.environ.get('t')
        last_spk = self.get_spike()
        if global_args.diff_spike == 0:
            last_spk = jax.lax.stop_gradient(last_spk)
        last_V = self.V.value - self.V_th_inf * last_spk
        last_I2 = self.I2.value - self.A2 * last_spk
        I2 = brainstate.nn.exp_euler_step(self.dI2, last_I2, t)
        V = brainstate.nn.exp_euler_step(self.dV, last_V, t, I_ext=(x + I2))
        self.I2.value = I2
        self.V.value = V
        # output
        inp = jax.nn.standardize(self.V.value - self.V_th_inf)
        return inp

    def get_spike(self, V=None):
        V = self.V.value if V is None else V
        spk = self.spk_fun((V - self.V_th_inf) / self.V_th_inf)
        return spk


class GifNet(brainstate.nn.Module):
    def __init__(
        self, num_in, num_rec, num_out, args, filepath: str = None
    ):
        super().__init__()

        self.filepath = filepath

        ff_init = brainstate.init.KaimingNormal(scale=args.ff_scale)
        rec_init = brainstate.init.KaimingNormal(scale=args.rec_scale)
        w = jnp.concatenate([ff_init((num_in, num_rec)), rec_init((num_rec, num_rec))], axis=0)
        ir2r = brainscale.nn.Linear(num_in + num_rec, num_rec, w_init=w)

        # parameters
        self.num_in = num_in
        self.num_rec = num_rec
        self.num_out = num_out
        self.ir2r = ir2r
        self.exp = brainscale.nn.Expon(num_rec, tau=args.tau_syn)
        tau_I2 = brainstate.random.uniform(100., args.tau_I2 * 1.5, num_rec)
        self.r = GIF(num_rec, V_rest=0., V_th_inf=1., spike_fun=brainstate.surrogate.ReluGrad(),
                     A2=args.A2, tau=args.tau_neu, tau_I2=tau_I2)
        self.out = brainscale.nn.LeakyRateReadout(
            num_rec, num_out, tau=args.tau_o,
            w_init=brainstate.init.KaimingNormal(scale=args.ff_scale)
        )

    def membrane_reg(self, mem_low: float, mem_high: float, factor: float = 0.):
        loss = 0.
        if factor > 0.:
            # extract all Neuron models
            neurons = self.nodes().subset(brainstate.nn.Neuron).unique().values()
            # evaluate the membrane potential
            for l in neurons:
                loss += jnp.square(jnp.mean(jax.nn.relu(l.V.value - mem_high) ** 2 +
                                            jax.nn.relu(mem_low - l.V.value) ** 2))
            loss = loss * factor
        return loss

    def spike_reg(self, target_fr: float, factor: float = 0.):
        # target_fr: Hz
        loss = 0.
        if factor > 0.:
            # extract all Neuron models
            neurons = self.nodes().subset(brainstate.nn.Neuron).unique().values()
            # evaluate the spiking dynamics
            for l in neurons:
                loss += (jnp.mean(l.get_spike()) - target_fr / 1e3 * brainstate.environ.get_dt()) ** 2
            loss = loss * factor
        return loss

    def to_state_dict(self):
        res = dict(
            ir2r=self.ir2r.weight_op.value,
            out=self.out.weight_op.value,
            tau_I2=self.r.tau_I2,
        )
        return jax.tree.map(np.asarray, res)

    def from_state_dict(self, state_dict: dict):
        state_dict = jax.tree.map(jnp.asarray, state_dict)
        self.ir2r.weight_op.value = state_dict['ir2r']
        self.out.weight_op.value = state_dict['out']
        self.r.tau_I2 = state_dict['tau_I2']

    def save(self, **kwargs):
        if self.filepath is not None:
            states = self.to_state_dict()
            states.update(kwargs)
            print(f'Saving the model to {self.filepath}/final_states.pkl ...')
            with open(f'{self.filepath}/final_states.pkl', 'wb') as f:
                pickle.dump(states, f)

    def restore(self):
        if self.filepath is not None:
            print(f'Loading the model from {self.filepath}/final_states.pkl ...')
            with open(f'{self.filepath}/final_states.pkl', 'rb') as f:
                states = pickle.load(f)
            self.from_state_dict(states)

    def update(self, spikes):
        cond = self.ir2r(jnp.concatenate([spikes, self.r.get_spike()], axis=-1))
        out = self.r(self.exp(cond))
        return self.out(out)

    @brainstate.compile.jit(static_argnums=0)
    def etrace_predict(self, inputs, targets):
        brainstate.nn.init_all_states(self, inputs.shape[1])

        def _single_step(i, inp, fit: bool = True):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt(), fit=fit):
                out = self.update(inp)
            return out

        # initialize the online learning model
        if global_args.method == 'expsm_diag':
            model = brainscale.IODimVjpAlgorithm(
                _single_step,
                int(global_args.etrace_decay) if global_args.etrace_decay > 1. else global_args.etrace_decay,
                vjp_method=global_args.vjp_time,
            )
            model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
        elif global_args.method == 'diag':
            model = brainscale.ParamDimVjpAlgorithm(
                _single_step,
                vjp_method=global_args.vjp_time,
            )
            model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
        elif global_args.method == 'hybrid':
            model = brainscale.HybridDimVjpAlgorithm(
                _single_step,
                int(global_args.etrace_decay) if global_args.etrace_decay > 1. else global_args.etrace_decay,
                vjp_method=global_args.vjp_time,
            )
            model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
        else:
            raise ValueError(f'Unknown online learning methods: {global_args.method}.')

        model.graph.show_graph()

        def _predict_step(i, inp):
            out = model(i, inp, running_index=i)
            etrace = model.get_etrace_of(self.ir2r.weight_op)
            if global_args.method == 'diag':
                etrace = jax.tree.map(lambda x: x[0], etrace)
            elif global_args.method == 'expsm_diag':
                etrace = jax.tree.map(lambda x: x[0], etrace)
            mem = self.r.V.value
            return out, etrace, mem

        outs, etraces, mems = brainstate.compile.for_loop(_predict_step, np.arange(inputs.shape[0]), inputs)
        n_sim = format_sim_epoch(global_args.warmup_ratio, inputs.shape[0])
        acc = jnp.mean(jnp.equal(targets, jnp.argmax(jnp.mean(outs[n_sim:], axis=0), axis=1)))
        return mems, acc, etraces

    def verify(self, dataloader, x_func, num_show=5, sps_inc=10., filepath=None):
        def _step(index, x):
            with brainstate.environ.context(i=index, t=index * brainstate.environ.get_dt()):
                out = self.update(x)
            return out, self.r.get_spike(), self.r.V.value

        dataloader = iter(dataloader)
        xs, ys = next(dataloader)  # xs: [n_samples, n_steps, n_in]
        xs = jnp.asarray(x_func(xs))
        print(xs.shape, ys.shape)
        brainstate.nn.init_all_states(self, xs.shape[1])

        time_indices = np.arange(0, xs.shape[0])
        outs, sps, vs = brainstate.compile.for_loop(_step, time_indices, xs)
        outs = u.math.as_numpy(outs)
        sps = u.math.as_numpy(sps)
        vs = u.math.as_numpy(vs)
        vs = np.where(sps, vs + sps_inc, vs)

        ts = time_indices * brainstate.environ.get_dt()
        max_t = xs.shape[0] * brainstate.environ.get_dt()

        for i in range(num_show):
            fig, gs = braintools.visualize.get_figure(4, 1, 2., 10.)

            ax_inp = fig.add_subplot(gs[0, 0])
            indices, times = raster_plot(xs[:, i], ts)
            ax_inp.plot(times, indices, '.')
            ax_inp.set_xlim(0., max_t)
            ax_inp.set_ylabel('Input Activity')

            ax = fig.add_subplot(gs[1, 0])
            plt.plot(ts, vs[:, i])
            # for j in range(0, self.r.num, 10):
            #   pass
            ax.set_xlim(0., max_t)
            ax.set_ylabel('Recurrent Potential')

            # spiking activity
            ax_rec = fig.add_subplot(gs[2, 0])
            indices, times = raster_plot(sps[:, i], ts)
            ax_rec.plot(times, indices, '.')
            ax_rec.set_xlim(0., max_t)
            ax_rec.set_ylabel('Recurrent Spiking')

            # decision activity
            ax_out = fig.add_subplot(gs[3, 0])
            for j in range(outs.shape[-1]):
                ax_out.plot(ts, outs[:, i, j], label=f'Readout {j}', alpha=0.7)
            ax_out.set_ylabel('Output Activity')
            ax_out.set_xlabel('Time [ms]')
            ax_out.set_xlim(0., max_t)
            plt.legend()

            if filepath:
                plt.savefig(f'{filepath}/{i}.png')

        if filepath is None:
            plt.show()
        plt.close()


class Trainer(object):
    """
    The training class with only loss.
    """

    def __init__(
        self,
        target: GifNet,
        opt: brainstate.optim.Optimizer,
        arguments: argparse.Namespace,
        filepath: str
    ):
        super().__init__()

        self.filepath = filepath
        self.file = None
        if filepath:
            if not os.path.exists(self.filepath):
                os.makedirs(self.filepath, exist_ok=True)
            self.file = open(f'{self.filepath}/loss.txt', 'w')

        # exponential smoothing
        self.smoother = ExponentialSmooth(0.8)

        # target network
        self.target = target

        # parameters
        self.args = arguments

        # optimizer
        self.opt = opt
        opt.register_trainable_weights(self.target.states().subset(brainstate.ParamState))

    def print(self, msg):
        print(msg)
        if self.file is not None:
            print(msg, file=self.file)

    def _acc(self, out, target):
        return jnp.mean(jnp.equal(target, jnp.argmax(jnp.mean(out, axis=0), axis=1)))

    def _loss(self, out, target):
        loss = braintools.metric.softmax_cross_entropy_with_integer_labels(out, target).mean()

        # L1 regularization loss
        if self.args.weight_L1 != 0.:
            leaves = self.target.states().subset(brainstate.ParamState).to_dict_values()
            loss += self.args.weight_L1 * reduce(jnp.add, jax.tree.map(lambda x: jnp.sum(jnp.abs(x)), leaves))

        # membrane potential regularization loss
        if self.args.v_reg_factor != 0.:
            mem_low = self.args.v_reg_low
            mem_high = self.args.v_reg_high
            loss += self.target.membrane_reg(mem_low, mem_high, self.args.v_reg_factor)

        # spike regularization loss
        if self.args.spk_reg_factor != 0.:
            fr = self.args.spk_reg_rate
            loss += self.target.spike_reg(fr, self.args.spk_reg_factor)

        return loss

    @brainstate.compile.jit(static_argnums=(0,))
    def etrace_train(self, inputs, targets):
        # initialize the states
        brainstate.nn.init_all_states(self.target, inputs.shape[1])
        # weights
        weights = self.target.states().subset(brainstate.ParamState)

        # the model for a single step
        def _single_step(i, inp, fit: bool = True):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt(), fit=fit):
                out = self.target(inp)
            return out

        # initialize the online learning model
        if self.args.method == 'expsm_diag':
            model = brainscale.IODimVjpAlgorithm(
                _single_step,
                int(self.args.etrace_decay) if self.args.etrace_decay > 1. else self.args.etrace_decay,
                vjp_method=self.args.vjp_time,
            )
            model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
        elif self.args.method == 'diag':
            model = brainscale.ParamDimVjpAlgorithm(
                _single_step,
                vjp_method=self.args.vjp_time,
            )
            model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
        elif self.args.method == 'hybrid':
            model = brainscale.HybridDimVjpAlgorithm(
                _single_step,
                int(self.args.etrace_decay) if self.args.etrace_decay > 1. else self.args.etrace_decay,
                vjp_method=self.args.vjp_time,
            )
            model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
        else:
            raise ValueError(f'Unknown online learning methods: {self.args.method}.')

        model.show_graph()

        def _etrace_grad(i, inp):
            # call the model
            out = model(i, inp, running_index=i)
            # calculate the loss
            loss = self._loss(out, targets)
            return loss, out

        def _etrace_step(prev_grads, x):
            # no need to return weights and states, since they are generated then no longer needed
            i, inp = x
            f_grad = brainstate.compile.grad(_etrace_grad, grad_vars=weights, has_aux=True, return_value=True)
            cur_grads, local_loss, out = f_grad(i, inp)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, (out, local_loss)

        def _etrace_train(indices_, inputs_):
            # forward propagation
            grads = jax.tree.map(jnp.zeros_like, weights.to_dict_values())
            grads, (outs, losses) = brainstate.compile.scan(_etrace_step, grads, (indices_, inputs_))
            # gradient updates
            grads = brainstate.functional.clip_grad_norm(grads, 1.)
            self.opt.update(grads)
            # accuracy
            return losses.mean(), outs

        # running indices
        indices = np.arange(inputs.shape[0])
        if self.args.warmup_ratio > 0:
            n_sim = format_sim_epoch(self.args.warmup_ratio, inputs.shape[0])
            brainstate.compile.for_loop(lambda i, inp: model(i, inp, running_index=i), indices[:n_sim], inputs[:n_sim])
            loss, outs = _etrace_train(indices[n_sim:], inputs[n_sim:])
        else:
            loss, outs = _etrace_train(indices, inputs)

        # returns
        return loss, self._acc(outs, targets)

    @brainstate.compile.jit(static_argnums=(0,))
    def bptt_train(self, inputs, targets):
        # running indices
        indices = np.arange(inputs.shape[0])
        # initialize the states
        brainstate.nn.init_all_states(self.target, inputs.shape[1])

        # the model for a single step
        def _single_step(i, inp, fit: bool = True):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt(), fit=fit):
                out = self.target(inp)
            return out

        def _run_step_train(i, inp):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                out = self.target(inp)
                loss = self._loss(out, targets)
            return out, loss

        def _bptt_grad_step():
            if self.args.warmup_ratio > 0:
                n_sim = format_sim_epoch(self.args.warmup_ratio, inputs.shape[0])
                _ = brainstate.compile.for_loop(_single_step, indices[:n_sim], inputs[:n_sim])
                outs, losses = brainstate.compile.for_loop(_run_step_train, indices[n_sim:], inputs[n_sim:])
            else:
                outs, losses = brainstate.compile.for_loop(_run_step_train, indices, inputs)
            return losses.mean(), outs

        # gradients
        weights = self.target.states().subset(brainstate.ParamState)
        grads, loss, outs = brainstate.augment.grad(_bptt_grad_step, grad_vars=weights, has_aux=True,
                                                    return_value=True)()

        # optimization
        # jax.debug.print('grads = {g}', g=jax.tree.map(jnp.max, grads))
        grads = brainstate.functional.clip_grad_norm(grads, 1.)
        self.opt.update(grads)

        return loss, self._acc(outs, targets)

    def f_train(self, train_loader, x_func, y_func):
        self.print(self.args)

        max_acc = 0.
        t0 = time.time()
        try:
            for i, (x_local, y_local) in enumerate(train_loader):
                if i >= self.args.epochs:
                    break

                # inputs and targets
                x_local = x_func(x_local)
                y_local = y_func(y_local)

                # training
                loss, acc = (self.bptt_train(x_local, y_local)
                             if self.args.method == 'bptt' else
                             self.etrace_train(x_local, y_local))
                t = time.time() - t0
                self.print(f'Batch {i:4d}, loss = {float(loss):.8f}, acc = {float(acc):.6f}, time = {t:.5f} s')
                if (i + 1) % 100 == 0:
                    self.opt.lr.step_epoch()

                # accuracy
                avg_acc = self.smoother(acc)
                if avg_acc > max_acc:
                    max_acc = avg_acc
                    if platform.platform().startswith('Linux'):
                        self.target.save(loss=loss, acc=acc)
                if max_acc > self.args.acc_th:
                    self.print(f'The training accuracy is greater than {self.args.acc_th * 100}%. Training is stopped.')
                    break
                t0 = time.time()
        finally:
            if self.file is not None:
                self.file.close()


class DMS(IterableDataset):
    """
    Delayed match-to-sample task.
    """
    times = ('dead', 'fixation', 'sample', 'delay', 'test')
    output_features = ('non-match', 'match')

    _rotate_choice = {
        '0': 0,
        '45': 1,
        '90': 2,
        '135': 3,
        '180': 4,
        '225': 5,
        '270': 6,
        '315': 7,
        '360': 8,
    }

    def __init__(
        self,
        dt=100.,
        t_fixation=500.,
        t_sample=500.,
        t_delay=1000.,
        t_test=500.,
        limits=(0., np.pi * 2),
        rotation_match='0',
        kappa=3.,
        bg_fr=1.,
        ft_motion=bd.cognitive.Feature(24, 100, 40.),
        mode: str = 'rate',
    ):
        super().__init__()
        self.dt = dt
        # time
        self.t_fixation = int(t_fixation / dt)
        self.t_sample = int(t_sample / dt)
        self.t_delay = int(t_delay / dt)
        self.t_test = int(t_test / dt)
        self.num_steps = self.t_fixation + self.t_sample + self.t_delay + self.t_test
        self._times = {
            'fixation': self.t_fixation,
            'sample': self.t_sample,
            'delay': self.t_delay,
            'test': self.t_test,
        }
        test_onset = self.t_fixation + self.t_sample + self.t_delay
        self._test_onset = test_onset
        self.test_time = slice(test_onset, test_onset + self.t_test)
        self.fix_time = slice(0, test_onset)
        self.sample_time = slice(self.t_fixation, self.t_fixation + self.t_sample)

        # input shape
        self.features = ft_motion.set_name('motion')
        self.features.set_mode(mode)
        self.rotation_match = rotation_match
        self._rotate = self._rotate_choice[rotation_match]
        self.bg_fr = bg_fr  # background firing rate
        self.v_min = limits[0]
        self.v_max = limits[1]
        self.v_range = limits[1] - limits[0]

        # Tuning function data
        self.n_motion_choice = 8
        self.kappa = kappa  # concentration scaling factor for von Mises

        # Generate list of preferred directions
        # dividing neurons by 2 since two equal
        # groups representing two modalities
        pref_dirs = np.arange(self.v_min, self.v_max, self.v_range / ft_motion.num)

        # Generate list of possible stimulus directions
        stim_dirs = np.arange(self.v_min, self.v_max, self.v_range / self.n_motion_choice)

        d = np.cos(np.expand_dims(stim_dirs, 1) - pref_dirs)
        self.motion_tuning = np.exp(self.kappa * d) / np.exp(self.kappa)

    @property
    def num_inputs(self) -> int:
        return self.features.num

    @property
    def num_outputs(self) -> int:
        return 2

    def sample_a_trial(self, *index):
        fr = self.features.fr(self.dt)
        bg_fr = bd.cognitive.firing_rate(self.bg_fr, self.dt, self.features.mode)
        return self._dms(self.num_steps,
                         self.num_inputs,
                         self.n_motion_choice,
                         self.motion_tuning,
                         self.features.is_spiking_mode,
                         self.sample_time,
                         self.test_time,
                         fr,
                         bg_fr,
                         self._rotate)

    @staticmethod
    @njit
    def _dms(num_steps, num_inputs, n_motion_choice, motion_tuning, is_spiking_mode,
             sample_time, test_time, fr, bg_fr, rotate_dir):
        # data
        X = np.zeros((num_steps, num_inputs))

        # sample
        match = np.random.randint(2)
        sample_dir = np.random.randint(n_motion_choice)

        # Generate the sample and test stimuli based on the rule
        if match == 1:  # match trial
            test_dir = (sample_dir + rotate_dir) % n_motion_choice
        else:
            test_dir = np.random.randint(n_motion_choice)
            while test_dir == ((sample_dir + rotate_dir) % n_motion_choice):
                test_dir = np.random.randint(n_motion_choice)

        # SAMPLE stimulus
        X[sample_time] += motion_tuning[sample_dir] * fr
        # TEST stimulus
        X[test_time] += motion_tuning[test_dir] * fr
        X += bg_fr

        # to spiking
        if is_spiking_mode:
            X = np.random.random(X.shape) < X
            X = X.astype(np.float32)

        # can use a greater weight for test period if needed
        return X, match

    def __iter__(self):
        while True:
            yield self.sample_a_trial()


def plot_weight_dist(weight_vals: dict, n_in, show: bool = True, title='', filepath=None, dist=True):
    fig, gs = braintools.visualize.get_figure(1, len(weight_vals), 3., 4.5)
    for i, (k, v) in enumerate(weight_vals.items()):
        ax = fig.add_subplot(gs[0, i])
        if isinstance(v, dict):
            v = v['weight']
            if dist:
                v1, v2 = jnp.split(v, [n_in], axis=0)
                ax.hist(v1.flatten(), bins=100, density=True, alpha=0.5, label='Input')
                ax.hist(v2.flatten(), bins=100, density=True, alpha=0.5, label='Recurrent')
                plt.legend()
            else:
                # print(jnp.linalg.matrix_rank(v))
                c = ax.pcolor(v, cmap='viridis')
                plt.colorbar(c, ax=ax)
        else:
            if dist:
                ax.hist(v.flatten(), bins=100, density=True, alpha=0.5)
            else:
                # print(jnp.linalg.matrix_rank(v))
                c = ax.pcolor(v, cmap='viridis')
                plt.colorbar(c, ax=ax)
        ax.set_title(k)
    if title:
        plt.suptitle(title)
    if filepath:
        plt.savefig(f'{filepath}/weight_dist-{title}.png')
    if show:
        plt.show()


def load_data():
    # loading the data
    task = DMS(dt=brainstate.environ.get_dt(), mode='spiking', bg_fr=1., t_fixation=global_args.t_fixation,
               t_sample=500.,
               t_delay=global_args.t_delay, t_test=500., ft_motion=bd.cognitive.Feature(24, 100, 100.))
    train_loader = DataLoader(task, batch_size=global_args.batch_size)
    input_process = lambda x: jnp.asarray(x, dtype=brainstate.environ.dftype()).transpose(1, 0, 2)
    label_process = lambda x: jnp.asarray(x, dtype=brainstate.environ.ditype())
    global_args.warmup_ratio = task.num_steps - task.t_test
    global_args.n_out = task.num_outputs
    n_in = task.num_inputs
    return train_loader, input_process, label_process, n_in


def load_a_batch_data():
    # loading the data
    task = DMS(dt=brainstate.environ.get_dt(), mode='spiking', bg_fr=1., t_fixation=global_args.t_fixation,
               t_sample=500.,
               t_delay=global_args.t_delay, t_test=500., ft_motion=bd.cognitive.Feature(24, 100, 100.))
    input_process = lambda x: jnp.asarray(x, dtype=brainstate.environ.dftype()).transpose(1, 0, 2)
    label_process = lambda x: jnp.asarray(x, dtype=brainstate.environ.ditype())
    global_args.warmup_ratio = task.num_steps - task.t_test
    global_args.n_out = task.num_outputs
    n_in = task.num_inputs

    xs = []
    ys = []
    for i in range(1):
        x, y = task.sample_a_trial()
        xs.append(x)
        ys.append(y)
    xs = input_process(xs)
    ys = label_process(ys)
    return xs, ys, n_in


def show_weight_difference():
    different_path = {
        'ES-D-RTRL': (
            r'results\rsnn-for-dms\diff-spike\expsm_diag t 0 decay=0.99\t_delay=1000.0-tau_I2=1500.0-tau_neu=100.0-tau_syn=100.0-A2=1.0-ffscale=6.0-recscale=2.0-diff_spike=1-2024-07-04 19-31-46'
        ),
        'BPTT': (
            r'results\rsnn-for-dms\diff-spike\bptt\t_delay=1000.0-tau_I2=1500.0-tau_neu=100.0-tau_syn=100.0-A2=1.0-ffscale=6.0-recscale=2.0-diff_spike=1-2024-07-04 19-31-29'
        ),
        'D-RTRL': (
            r'results\rsnn-for-dms\diff-spike\diag t 0\t_delay=1000.0-tau_I2=1500.0-tau_neu=100.0-tau_syn=100.0-A2=1.0-ffscale=6.0-recscale=2.0-diff_spike=1-2024-07-04 19-13-42'
        ),
    }

    # different_path = {
    #   'ES-D-RTRL': (
    #     r'results\rsnn-for-dms\non-diff-spike\expsm_diag t 0 decay=0.99\t_delay=1000.0-tau_I2=1500.0-tau_neu=100.0-tau_syn=100.0-A2=1.0-ffscale=6.0-recscale=2.0-diff_spike=0-2024-07-04 19-33-39'
    #   ),
    #   'BPTT': (
    #     r'results\rsnn-for-dms\non-diff-spike\bptt\t_delay=1000.0-tau_I2=1500.0-tau_neu=100.0-tau_syn=100.0-A2=1.0-ffscale=6.0-recscale=2.0-diff_spike=0-2024-07-04 19-26-03'
    #   ),
    #   'D-RTRL': (
    #     r'results\rsnn-for-dms\non-diff-spike\diag t 0\t_delay=1000.0-tau_I2=1500.0-tau_neu=100.0-tau_syn=100.0-A2=1.0-ffscale=6.0-recscale=2.0-diff_spike=0-2024-07-04 19-24-01'
    #   ),
    # }

    def visualize_weight(w, title=''):
        w_in, w_rec = np.split(w, [n_in], axis=0)
        fig, gs = braintools.visualize.get_figure(1, 2, 3., 3.5)
        ax = fig.add_subplot(gs[0, 0])
        c = ax.pcolor(w_in, cmap='viridis')
        plt.colorbar(c, ax=ax)
        plt.xticks([])
        plt.yticks([])
        plt.title('$W_\mathrm{in}$')
        ax = fig.add_subplot(gs[0, 1])
        c = ax.pcolor(w_rec, cmap='viridis')
        plt.colorbar(c, ax=ax)
        plt.xticks([])
        plt.yticks([])
        plt.title('$W_\mathrm{rec}$')
        if title:
            plt.suptitle(title)
        # plt.show()

    def visualize_weight_dist(w, title=''):
        w_in, w_rec = np.split(w, [n_in], axis=0)
        fig, gs = braintools.visualize.get_figure(1, 2, 3., 3.5)
        ax = fig.add_subplot(gs[0, 0])
        plt.hist(w_in.flatten(), bins=100, density=True)
        plt.xlabel('Weight Value')
        plt.ylabel('Density')
        plt.title('$W_\mathrm{in}$')
        ax = fig.add_subplot(gs[0, 1])
        plt.hist(w_rec.flatten(), bins=100, density=True)
        plt.xlabel('Weight Value')
        plt.title('$W_\mathrm{rec}$')
        if title:
            plt.suptitle(title)
        # plt.show()

    def pca_visualize(w, title=''):
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        pca.fit(w)
        w_pca = pca.transform(w)
        fig, gs = braintools.visualize.get_figure(1, 1, 3., 4.5)
        ax = fig.add_subplot(gs[0, 0])
        scatter = plt.scatter(w_pca[:, 0], w_pca[:, 1])
        # cbar = plt.colorbar(scatter, ax=ax)
        # cbar.set_label('Labels')
        plt.xlim(-2.0, 2.0)
        plt.ylim(-2.0, 2.0)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        ax.grid(True)  # Add grid lines
        if title:
            plt.suptitle(title)

    def stability_plot(w, title=''):
        w_in, w_rec = np.split(w, [n_in], axis=0)
        fig, gs = braintools.visualize.get_figure(1, 1, 3., 4.5)

        ax = fig.add_subplot(gs[0, 0])
        eigen_vals = np.linalg.eigvals(w_rec)
        plt.scatter(eigen_vals.real, eigen_vals.imag)
        circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--')
        plt.gca().add_artist(circle)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axis('equal')
        if title:
            plt.suptitle(title)
        # plt.show()

    global global_args
    for key, filepath in different_path.items():
        with open(f'{filepath}/loss.txt', 'r') as f:
            print(f'Loading {filepath} ...')

            args = f.readline().strip().replace('Namespace', 'dict')
            global_args = brainstate.util.DotDict(eval(args))
            global_args['t_fixation'] = 500.

            brainstate.environ.set(
                mode=brainstate.mixin.JointMode(brainstate.mixin.Batching(), brainstate.mixin.Training()),
                dt=global_args.dt
            )
            brainstate.util.clear_name_cache()
            train_loader, input_process, label_process, n_in = load_data()
            net = GifNet(n_in, global_args.n_rec, global_args.n_out, global_args, filepath=filepath)
            net.restore()
            weight = net.ir2r.weight_op.value['weight']

            visualize_weight(weight, title=key)
            # visualize_weight_dist(weight, title=key)
            # pca_visualize(weight.T, title=key)
            # stability_plot(weight, title=key)

    plt.show()


@njit
def numba_seed(seed):
    np.random.seed(seed)


def show_expsm_diag_etrace():
    filepath = (
        r'results\rsnn-for-dms\diff-spike\expsm_diag t 0 decay=0.99\t_delay=1000.0-tau_I2=1500.0-tau_neu=100.0-tau_syn=100.0-A2=1.0-ffscale=6.0-recscale=2.0-diff_spike=1-2024-07-04 19-31-46')
    # filepath = (r'results\rsnn-for-dms\non-diff-spike\expsm_diag t 0 decay=0.99\t_delay=1000.0-tau_I2=1500.0-tau_neu=100.0-tau_syn=100.0-A2=1.0-ffscale=6.0-recscale=2.0-diff_spike=0-2024-07-04 19-33-39')

    global global_args
    with open(f'{filepath}/loss.txt', 'r') as f:
        print(f'Loading {filepath} ...')

        seed = 2024
        brainstate.random.seed(seed)
        numba_seed(seed)

        args = f.readline().strip().replace('Namespace', 'dict')
        global_args = brainstate.util.DotDict(eval(args))
        global_args['t_fixation'] = 500.

        brainstate.environ.set(
            mode=brainstate.mixin.JointMode(brainstate.mixin.Batching(), brainstate.mixin.Training()),
            dt=global_args.dt
        )
        brainstate.util.clear_name_cache()
        xs, ys, n_in = load_a_batch_data()
        indices = np.arange(xs.shape[0])
        net = GifNet(n_in, global_args.n_rec, global_args.n_out, global_args, filepath=filepath)
        net.restore()
        mems, acc, etraces = net.etrace_predict(xs, ys)
        print(acc)

        # n_to_vis = 10
        # fig, gs = bts.visualize.get_figure(len(etraces[0]) + len(etraces[1]), 1, 2., 10.)
        # for i, val in enumerate(etraces[0].values()):
        #   ax = fig.add_subplot(gs[i, 0])
        #   plt.plot(indices, val[:, 0, :n_to_vis])
        # for j, val in enumerate(etraces[1].values()):
        #   ax = fig.add_subplot(gs[len(etraces[0]) + j, 0])
        #   plt.plot(indices, val[:, 0, :n_to_vis])
        # plt.show()

        # num = 3
        # fig, gs = bts.visualize.get_figure(len(etraces[0]) + len(etraces[1]), num, 2., 4.)
        # for k in range(num):
        #   for i, val in enumerate(etraces[0].values()):
        #     ax = fig.add_subplot(gs[i, k])
        #     plt.plot(indices, val[:, k])
        #   n = len(etraces[0])
        #   for j, val in enumerate(etraces[1].values()):
        #     ax = fig.add_subplot(gs[n + j, k])
        #     plt.plot(indices, val[:, k])
        #   # ax = fig.add_subplot(gs[-1, k])
        #   # plt.plot(indices, mems[:, 0, k])
        # plt.show()

        def plot_other(min_, max_, name):
            plt.ylabel(name)
            plt.axvline(500., linestyle='--')
            plt.axvline(1000., linestyle='--')
            plt.axvline(2000., linestyle='--')

            plt.fill_between([0., 500.], min_, max_, color='ivory', )
            plt.fill_between([500., 1000.], min_, max_, color='linen', )
            plt.fill_between([1000., 2000.], min_, max_, color='ivory', )
            plt.fill_between([2000., 2500.], min_, max_, color='linen', )

            plt.xlim(0, xs.shape[0])

        plt.style.use(['science', 'nature', 'notebook'])
        n_gs = len(etraces[0]) + len(etraces[1])
        fig, gs = braintools.visualize.get_figure(n_gs, 1, 6.0 / n_gs, 9.0)
        names = ['$\epsilon_\mathrm{pre}$', '$\epsilon_\mathrm{post,g}$',
                 '$\epsilon_\mathrm{post,a}$', '$\epsilon_\mathrm{post,V}$', ]
        axes = []

        n_to_vis = 3
        for i, val in enumerate(etraces[0].values()):
            min_, max_ = np.inf, 0.
            ax = fig.add_subplot(gs[i, 0])
            axes.append(ax)
            for k in np.random.randint(0, 100, n_to_vis):
                data = val[:, k]
                min_ = min(min_, data.min())
                max_ = max(max_, data.max())
                plt.plot(indices, data, label='$\mathrm{pre}_{%d}$' % k)
            plt.xticks([])
            plt.legend(fontsize=10, ncol=1, bbox_to_anchor=(1.01, 1.01))
            plot_other(min_, max_, names[i])
        i_start = len(etraces[0])

        matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['#e41a1c', '#a65628', '#984ea3'])
        # matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
        # color=[  # '#377eb8', '#ff7f00', '#4daf4a',
        #     '#f781bf', '#a65628', '#984ea3',
        #     '#999999', '#e41a1c', '#dede00']
        # )
        # matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
        #   color=['peru', 'darkviolet', 'darkcyan', 'darkorange',
        #   'darkgreen', 'darkred', 'darkblue', 'darkmagenta', 'darkgoldenrod' ]
        # )
        neu_post_ids = np.random.randint(0, 100, n_to_vis)
        for j, val in enumerate(etraces[1].values()):
            ax = fig.add_subplot(gs[i_start + j, 0])
            axes.append(ax)
            min_, max_ = np.inf, 0.
            for k in neu_post_ids:
                data = val[:, k]
                min_ = min(min_, data.min())
                max_ = max(max_, data.max())
                ax.plot(indices, data, label='$\mathrm{post}_{%d}$' % k)
            plt.legend(fontsize=10, ncol=1, bbox_to_anchor=(1.01, 1.01))
            plot_other(min_, max_, names[i_start + j])
            if j == len(etraces[1]) - 1:
                plt.xlabel('Time [ms]')
            else:
                plt.xticks([])

        # ax = fig.add_subplot(gs[-1, 0])
        # axes.append(ax)
        # min_, max_ = np.inf, 0.
        # for k in neu_post_ids:
        #   data = mems[:, 0, k]
        #   min_ = min(min_, data.min())
        #   max_ = max(max_, data.max())
        #   plt.plot(indices, data, label='$\mathrm{post}_{%d}$' % k)
        # plt.legend(fontsize=10, ncol=1, bbox_to_anchor=(1.001, 1.01))
        # plot_other(min_, max_, 'Potential')

        fig.align_ylabels(axes)
        # plt.suptitle('ETrace in D-RTRL', fontsize=16)
        plt.show()


def show_diag_etrace():
    filepath = r'results\rsnn-for-dms\diff-spike\diag t 0\t_delay=1000.0-tau_I2=1500.0-tau_neu=100.0-tau_syn=100.0-A2=1.0-ffscale=6.0-recscale=2.0-diff_spike=1-2024-07-04 19-13-42'
    # filepath = r'results\rsnn-for-dms\non-diff-spike\diag t 0\t_delay=1000.0-tau_I2=1500.0-tau_neu=100.0-tau_syn=100.0-A2=1.0-ffscale=6.0-recscale=2.0-diff_spike=0-2024-07-04 19-24-01'

    global global_args
    with open(f'{filepath}/loss.txt', 'r') as f:
        print(f'Loading {filepath} ...')

        args = f.readline().strip().replace('Namespace', 'dict')
        global_args = brainstate.util.DotDict(eval(args))
        global_args['t_fixation'] = 500.

        seed = 2024
        brainstate.random.seed(seed)
        numba_seed(seed)

        brainstate.environ.set(
            mode=brainstate.mixin.JointMode(brainstate.mixin.Batching(), brainstate.mixin.Training()),
            dt=global_args.dt
        )
        brainstate.util.clear_name_cache()
        # dataset
        xs, ys, n_in = load_a_batch_data()
        indices = np.arange(xs.shape[0])
        # network
        net = GifNet(n_in, global_args.n_rec, global_args.n_out, global_args, filepath=filepath)
        net.restore()
        # prediction
        mems, acc, etraces = net.etrace_predict(xs, ys)
        print(acc)

        plt.style.use(['science', 'nature', 'notebook'])
        fig, gs = braintools.visualize.get_figure(len(etraces), 1, 6 // len(etraces), 9.0)
        names = ['$\epsilon_g$', '$\epsilon_{a}$', '$\epsilon_{V}$', ]
        axes = []
        for i, val in enumerate(etraces.values()):
            min_, max_ = np.inf, 0.

            neuron_ids = [131, 160, 122, 57, 31]
            ax = fig.add_subplot(gs[i, 0])
            axes.append(ax)
            for i_neuron in neuron_ids:
                data = val['weight'][:, i_neuron, 0]
                min_ = min(min_, data.min())
                max_ = max(max_, data.max())
                plt.plot(indices, data, label=f'(0, {i_neuron})')

            if i == 0:
                plt.legend(fontsize=10, ncol=1, bbox_to_anchor=(1.01, 1.05))
            if i + 1 == len(etraces):
                plt.xlabel('Time [ms]')
            else:
                plt.xticks([])
            plt.ylabel(names[i])
            plt.axvline(500., linestyle='--')
            plt.axvline(1000., linestyle='--')
            plt.axvline(2000., linestyle='--')

            plt.fill_between([0., 500.], min_, max_, color='ivory', )
            plt.fill_between([500., 1000.], min_, max_, color='linen', )
            plt.fill_between([1000., 2000.], min_, max_, color='ivory', )
            plt.fill_between([2000., 2500.], min_, max_, color='linen', )

            plt.xlim(0, xs.shape[0])

        fig.align_ylabels(axes)
        # plt.suptitle('ETrace in D-RTRL', fontsize=16)
        plt.show()

        # num = 5
        # fig, gs = bts.visualize.get_figure(len(etraces), num, 2., 4.)
        # # neuron_ids = np.random.randint(0, global_args.n_rec, num)
        # print(neuron_ids)
        # for k in range(num):
        #   i_neuron = neuron_ids[k]
        #   for i, val in enumerate(etraces.values()):
        #     ax = fig.add_subplot(gs[i, k])
        #     plt.plot(indices, val['weight'][:, i_neuron, 0])
        # plt.show()


def network_training():
    # environment setting
    brainstate.environ.set(dt=global_args.dt)

    # get file path to output
    if global_args.filepath:
        filepath = global_args.filepath
    else:
        aim = f'results/rsnn-for-dms/'
        if global_args.exp_name:
            aim += global_args.exp_name
        now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
        param = (
            f't_delay={global_args.t_delay}-'
            f'tau_I2={global_args.tau_I2}-'
            f'tau_neu={global_args.tau_neu}-'
            f'tau_syn={global_args.tau_syn}-'
            f'A2={global_args.A2}-'
            f'ffscale={global_args.ff_scale}-'
            f'recscale={global_args.rec_scale}-'
            f'diff_spike={global_args.diff_spike}-{now}'
        )
        if global_args.method == 'bptt':
            filepath = f'{aim}/{global_args.method}/{param}'
        elif global_args.method == 'diag':
            filepath = f'{aim}/{global_args.method} {global_args.vjp_time}/{param}'
        else:
            filepath = (
                f'{aim}/{global_args.method} {global_args.vjp_time} '
                f'decay={global_args.etrace_decay}/{param}'
            )
    # filepath = None
    print(filepath)

    # loading the data
    train_loader, input_process, label_process, n_in = load_data()

    # creating the network and optimizer
    net = GifNet(n_in, global_args.n_rec, global_args.n_out, global_args, filepath=filepath)

    if global_args.mode == 'sim':
        if global_args.filepath:
            net.restore()
        net.verify(train_loader, input_process, num_show=5, sps_inc=10.)

    elif global_args.mode == 'train':
        opt = brainstate.optim.Adam(lr=global_args.lr, weight_decay=global_args.weight_L2)

        # creating the trainer
        trainer = Trainer(net, opt, global_args, filepath)
        trainer.f_train(
            train_loader,
            x_func=input_process,
            y_func=label_process,
        )

    else:
        raise ValueError(f'Unknown mode: {global_args.mode}')


if __name__ == '__main__':
    pass
    # network_training()
    # show_weight_difference()
    # show_expsm_diag_etrace()
    show_diag_etrace()
    # show_weight_difference()

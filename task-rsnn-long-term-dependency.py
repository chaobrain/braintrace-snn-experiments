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
import platform
import time
from functools import reduce
from typing import Optional, Any, Dict, Callable, Union, Iterator

import numpy as np

from utils import MyArgumentParser

parser = MyArgumentParser()

# Learning parameters
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs.")
parser.add_argument("--dt", type=float, default=1., help="The simulation time step.")
parser.add_argument("--loss", type=str, default='cel', choices=['cel', 'mse'], help="Loss function.")

# dataset
parser.add_argument("--mode", type=str, default="train", choices=['train', 'sim'], help="The computing mode.")
parser.add_argument("--dataset", type=str, default="dms", help="Choose between different datasets")
parser.add_argument("--n_data_worker", type=int, default=1, help="Number of data loading workers (default: 4)")
parser.add_argument("--t_delay", type=float, default=1e3, help="Deta delay length.")
parser.add_argument("--t_fixation", type=float, default=500., help="")

# training parameters
parser.add_argument("--exp_name", type=str, default='', help="")
parser.add_argument("--spk_fun", type=str, default='s2nn', help="spike surrogate gradient function.")
parser.add_argument("--warmup_ratio", type=float, default=0.0, help="The ratio for network simulation.")
parser.add_argument("--optimizer", type=str, default='adam', help="")
parser.add_argument("--acc_th", type=float, default=0.95, help="")
parser.add_argument("--filepath", type=str, default='', help="The name for the current experiment.")
parser.add_argument("--lr_milestones", type=str, default='', help="The name for the current experiment.")

# regularization parameters
parser.add_argument("--spk_reg_factor", type=float, default=0.0, help="Spike regularization factor.")
parser.add_argument("--spk_reg_rate", type=float, default=10., help="Target firing rate.")
parser.add_argument("--v_reg_factor", type=float, default=0.0, help="Voltage regularization factor.")
parser.add_argument("--v_reg_low", type=float, default=-20., help="The lowest voltage for regularization.")
parser.add_argument("--v_reg_high", type=float, default=1.4, help="The highest voltage for regularization.")
parser.add_argument("--weight_L1", type=float, default=0.0, help="The weight L1 regularization.")
parser.add_argument("--weight_L2", type=float, default=0.0, help="The weight L2 regularization.")

# GIF parameters
parser.add_argument("--detach_spk", type=int, default=0, help="Number of recurrent neurons.")
parser.add_argument("--n_rec", type=int, default=200, help="Number of recurrent neurons.")
parser.add_argument("--A2", type=float, default=-1.)
parser.add_argument("--tau_I2", type=float, default=2000.)
parser.add_argument("--tau_syn", type=float, default=10.)
parser.add_argument("--tau_o", type=float, default=10.)
parser.add_argument("--ff_scale", type=float, default=10.)
parser.add_argument("--rec_scale", type=float, default=2.)

global_args = parser.parse_args()

import matplotlib

if platform.platform().startswith('Linux'):
  matplotlib.use('agg')

import matplotlib.pyplot as plt
import brainscale as bnn
import brainstate as bst
import braintools as bts
import brainunit as bu
import brainpy_datasets as bd
import jax
import jax.numpy as jnp
import orbax.checkpoint
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


class Checkpointer(orbax.checkpoint.CheckpointManager):
  def __init__(
      self,
      directory: str,
      max_to_keep: Optional[int] = None,
      save_interval_steps: int = 1,
      metadata: Optional[Dict[str, Any]] = None,
  ):
    options = orbax.checkpoint.CheckpointManagerOptions(
      max_to_keep=max_to_keep,
      save_interval_steps=save_interval_steps,
      create=True
    )
    super().__init__(os.path.abspath(directory), options=options, metadata=metadata)

  def save_data(self, args: PyTree, step: int, **kwargs):
    return super().save(step, args=orbax.checkpoint.args.StandardSave(args), **kwargs)

  def load_data(self, args: PyTree, step: int = None, **kwargs):
    self.wait_until_finished()
    step = self.latest_step() if step is None else step
    tree = jax.tree_util.tree_map(orbax.checkpoint.utils.to_shape_dtype_struct, args)
    args = orbax.checkpoint.args.StandardRestore(tree)
    return super().restore(step, args=args, **kwargs)


class GIF(bst.nn.Neuron):
  def __init__(
      self, size,
      V_rest=0., V_th_inf=1., R=1., tau=20., tau_I2=50., A2=0.,
      V_initializer: Callable = bst.init.Constant(1.),
      I2_initializer: Callable = bst.init.ZeroInit(),
      spike_fun: Callable = bst.surrogate.ReluGrad(),
      spk_reset: str = 'soft',
      keep_size: bool = False,
      name: str = None,
      mode: bst.mixin.Mode = None,
      detach_spk: bool = False
  ):
    super().__init__(size, keep_size=keep_size, name=name, mode=mode,
                     spk_fun=spike_fun, spk_reset=spk_reset, detach_spk=detach_spk)

    # params
    self.V_rest = bst.init.param(V_rest, self.varshape, allow_none=False)
    self.V_th_inf = bst.init.param(V_th_inf, self.varshape, allow_none=False)
    self.R = bst.init.param(R, self.varshape, allow_none=False)
    self.tau = bst.init.param(tau, self.varshape, allow_none=False)
    self.tau_I2 = bst.init.param(tau_I2, self.varshape, allow_none=False)
    self.A2 = bst.init.param(A2, self.varshape, allow_none=False)

    # initializers
    self._V_initializer = V_initializer
    self._I2_initializer = I2_initializer

  def init_state(self, batch_size=None):
    self.V = bnn.ETraceVar(bst.init.param(self._V_initializer, self.varshape, batch_size))
    self.I2 = bnn.ETraceVar(bst.init.param(self._I2_initializer, self.varshape, batch_size))

  def dI2(self, I2, t):
    return - I2 / self.tau_I2

  def dV(self, V, t, I_ext):
    return (- V + self.V_rest + self.R * I_ext) / self.tau

  def update(self, x=0.):
    t = bst.environ.get('t')
    last_spk = self.get_spike()
    last_spk = jax.lax.stop_gradient(last_spk) if self.detach_spk else last_spk
    last_V = self.V.value - self.V_th_inf * last_spk
    last_I2 = self.I2.value + self.A2 * last_spk
    I2 = bst.nn.exp_euler_step(self.dI2, last_I2, t)
    V = bst.nn.exp_euler_step(self.dV, last_V, t, I_ext=(x + I2))
    self.I2.value = I2
    self.V.value = V
    return self.get_spike()

  def get_spike(self, V=None):
    V = self.V.value if V is None else V
    return self.spk_fun((V - self.V_th_inf) / self.V_th_inf)


class GifNet(bst.Module):
  def __init__(
      self, num_in, num_rec, num_out, args, filepath: str = None
  ):
    super().__init__()

    ff_init = bst.init.KaimingNormal(scale=args.ff_scale)
    rec_init = bst.init.KaimingNormal(scale=args.rec_scale)
    w = jnp.concatenate([ff_init((num_in, num_rec)), rec_init((num_rec, num_rec))], axis=0)
    ir2r = bnn.Linear(num_in + num_rec, num_rec, w_init=w)

    # parameters
    self.num_in = num_in
    self.num_rec = num_rec
    self.num_out = num_out
    self.ir2r = ir2r
    self.exp = bnn.Expon(num_rec, tau=args.tau_syn)
    tau_I2 = jnp.concat(
      [bst.random.uniform(1., 5., (num_rec // 2,)),
       bst.random.uniform(args.tau_I2 * 0.5, args.tau_I2 * 1.5, (num_rec // 2,))],
      axis=0
    )
    if args.spk_fun == 's2nn':
      spike_fun = bst.surrogate.S2NN()
    elif args.spk_fun == 's2nn2':
      spike_fun = bst.surrogate.S2NN(alpha=8., beta=1.)
    elif args.spk_fun == 'relu':
      spike_fun = bst.surrogate.ReluGrad()
    elif args.spk_fun == 'leaky_relu':
      spike_fun = bst.surrogate.LeakyRelu()
    elif args.spk_fun == 'inv_square':
      spike_fun = bst.surrogate.InvSquareGrad(10.)
    elif args.spk_fun == 'multi_gaussian':
      spike_fun = bst.surrogate.MultiGaussianGrad()
    else:
      raise ValueError
    self.r = GIF(num_rec, V_rest=0., V_th_inf=1., spike_fun=spike_fun, A2=args.A2, tau_I2=tau_I2,
                 detach_spk=bool(args.detach_spk))
    self.out = bnn.LeakyRateReadout(
      num_rec, num_out, tau=args.tau_o,
      w_init=bst.init.KaimingNormal(scale=args.ff_scale)
    )
    if filepath is None:
      self.checkpointer = None
    else:
      self.checkpointer = Checkpointer(filepath, max_to_keep=5)

  def membrane_reg(self, mem_low: float, mem_high: float, factor: float = 0.):
    loss = 0.
    if factor > 0.:
      # extract all Neuron models
      neurons = self.nodes().subset(bst.nn.Neuron).unique().values()
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
      neurons = self.nodes().subset(bst.nn.Neuron).unique().values()
      # evaluate the spiking dynamics
      for l in neurons:
        loss += (jnp.mean(l.get_spike()) - target_fr / 1e3 * bst.environ.get_dt()) ** 2
      loss = loss * factor
    return loss

  def save(self, step, **kwargs):
    if self.checkpointer is not None:
      states = self.states().subset(bst.ParamState).to_dict_values()
      self.checkpointer.save_data(states, step, **kwargs)

  def restore(self, step=None):
    if self.checkpointer is not None:
      states = self.states().subset(bst.ParamState)
      values = self.checkpointer.load_data(states.to_dict_values(), step=step)
      for k, v in values.items():
        states[k].value = v

  def update(self, spikes):
    cond = self.ir2r(jnp.concatenate([spikes, self.r.get_spike()], axis=-1))
    ext = self.exp(cond)
    return self.out(self.r(ext))

  def visualize_variables(self) -> dict:
    neurons = tuple(self.nodes().subset(bst.nn.Neuron).unique().values())
    outs = {
      'out_v': self.out.r.value,
      'rec_v': [l.V.value for l in neurons],
      'rec_s': [l.get_spike() for l in neurons],
    }
    return outs

  def verify(self, dataloader, x_func, num_show=5, sps_inc=10., filepath=None):
    def _step(index, x):
      with bst.environ.context(i=index, t=index * bst.environ.get_dt()):
        out = self.update(x)
      return out, self.r.get_spike(), self.r.V.value

    dataloader = iter(dataloader)
    xs, ys = next(dataloader)  # xs: [n_samples, n_steps, n_in]
    xs = jnp.asarray(x_func(xs))
    print(xs.shape, ys.shape)
    bst.init_states(self, xs.shape[1])

    time_indices = np.arange(0, xs.shape[0])
    outs, sps, vs = bst.transform.for_loop(_step, time_indices, xs)
    outs = bu.math.as_numpy(outs)
    sps = bu.math.as_numpy(sps)
    vs = bu.math.as_numpy(vs)
    vs = np.where(sps, vs + sps_inc, vs)

    ts = time_indices * bst.environ.get_dt()
    max_t = xs.shape[0] * bst.environ.get_dt()

    for i in range(num_show):
      fig, gs = bts.visualize.get_figure(4, 1, 2., 10.)

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
      opt: bst.optim.Optimizer,
      arguments: argparse.Namespace,
      filepath: str
  ):
    super().__init__()

    self.filepath = filepath
    if not os.path.exists(self.filepath):
      os.makedirs(self.filepath, exist_ok=True)
    self.file = open(f'{self.filepath}/loss.txt', 'w')

    # exponential smoothing
    self.smoother = ExponentialSmooth(0.8)

    # target network
    self.target = target

    # parameters
    self.args = arguments

    # loss function
    if self.args.loss == 'mse':
      self.loss_fn = bts.metric.squared_error
    elif self.args.loss == 'cel':
      self.loss_fn = bts.metric.softmax_cross_entropy_with_integer_labels
    else:
      raise ValueError

    # optimizer
    self.opt = opt
    opt.register_trainable_weights(self.target.states().subset(bst.ParamState))

  def print(self, msg):
    print(msg)
    print(msg, file=self.file)

  def _acc(self, out, target):
    return jnp.mean(jnp.equal(target, jnp.argmax(jnp.mean(out, axis=0), axis=1)))

  def _loss(self, out, target):
    loss = self.loss_fn(out, target).mean()

    # L1 regularization loss
    if self.args.weight_L1 != 0.:
      leaves = self.target.states().subset(bst.ParamState).to_dict_values()
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

  @bst.transform.jit(static_argnums=(0,))
  def etrace_train(self, inputs, targets):
    # initialize the states
    bst.init_states(self.target, inputs.shape[1])
    # weights
    weights = self.target.states().subset(bst.ParamState)

    # the model for a single step
    def _single_step(i, inp, fit: bool = True):
      with bst.environ.context(i=i, t=i * bst.environ.get_dt(), fit=fit):
        out = self.target(inp)
      return out

    # initialize the online learning model
    if self.args.method == 'expsm_diag':
      model = bnn.DiagIODimAlgorithm(
        _single_step,
        self.args.etrace_decay,
        num_snap=self.args.num_snap,
        snap_freq=self.args.snap_freq,
        diag_jacobian=self.args.diag_jacobian,
        diag_normalize=diag_norm_mapping[self.args.diag_normalize],
        vjp_time=self.args.vjp_time,
      )
      model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
    elif self.args.method == 'diag':
      model = bnn.DiagParamDimAlgorithm(
        _single_step,
        diag_jacobian=self.args.diag_jacobian,
        diag_normalize=diag_norm_mapping[self.args.diag_normalize],
        vjp_time=self.args.vjp_time,
      )
      model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
    elif self.args.method == 'hybrid':
      model = bnn.DiagHybridDimAlgorithm(
        _single_step,
        self.args.etrace_decay,
        num_snap=self.args.num_snap,
        snap_freq=self.args.snap_freq,
        diag_jacobian=self.args.diag_jacobian,
        diag_normalize=diag_norm_mapping[self.args.diag_normalize],
        vjp_time=self.args.vjp_time,
      )
      model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
    else:
      raise ValueError(f'Unknown online learning methods: {self.args.method}.')

    # model.graph.show_graph(start_frame=0, n_frame=5)

    def _etrace_grad(i, inp):
      # call the model
      out = model(i, inp, running_index=i)
      # calculate the loss
      loss = self._loss(out, targets)
      return loss, out

    def _etrace_step(prev_grads, x):
      # no need to return weights and states, since they are generated then no longer needed
      i, inp = x
      f_grad = bst.transform.grad(_etrace_grad, grad_vars=weights, has_aux=True, return_value=True)
      cur_grads, local_loss, out = f_grad(i, inp)
      next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
      return next_grads, (out, local_loss)

    def _etrace_train(indices_, inputs_):
      # forward propagation
      grads = jax.tree.map(lambda a: jnp.zeros_like(a), weights.to_dict_values())
      grads, (outs, losses) = bst.transform.scan(_etrace_step, grads, (indices_, inputs_))
      # gradient updates
      # jax.debug.print('grads = {g}', g=jax.tree.map(lambda a: jnp.max(jnp.abs(a)), grads))
      grads = bst.functional.clip_grad_norm(grads, 1.)
      self.opt.update(grads)
      # accuracy
      return losses.mean(), outs

    # running indices
    indices = np.arange(inputs.shape[0])
    if self.args.warmup_ratio > 0:
      n_sim = format_sim_epoch(self.args.warmup_ratio, inputs.shape[0])
      bst.transform.for_loop(lambda i, inp: model(i, inp, running_index=i), indices[:n_sim], inputs[:n_sim])
      loss, outs = _etrace_train(indices[n_sim:], inputs[n_sim:])
    else:
      loss, outs = _etrace_train(indices, inputs)

    # returns
    return loss, self._acc(outs, targets)

  @bst.transform.jit(static_argnums=(0,))
  def bptt_train(self, inputs, targets):
    # running indices
    indices = np.arange(inputs.shape[0])
    # initialize the states
    bst.init_states(self.target, inputs.shape[1])

    # the model for a single step
    def _single_step(i, inp, fit: bool = True):
      with bst.environ.context(i=i, t=i * bst.environ.get_dt(), fit=fit):
        out = self.target(inp)
      return out

    def _run_step_train(i, inp):
      with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
        out = self.target(inp)
        loss = self._loss(out, targets)
      return out, loss

    def _bptt_grad_step():
      if self.args.warmup_ratio > 0:
        n_sim = format_sim_epoch(self.args.warmup_ratio, inputs.shape[0])
        _ = bst.transform.for_loop(_single_step, indices[:n_sim], inputs[:n_sim])
        outs, losses = bst.transform.for_loop(_run_step_train, indices[n_sim:], inputs[n_sim:])
      else:
        outs, losses = bst.transform.for_loop(_run_step_train, indices, inputs)
      return losses.mean(), outs

    # gradients
    weights = self.target.states().subset(bst.ParamState)
    grads, loss, outs = bst.transform.grad(_bptt_grad_step, grad_vars=weights, has_aux=True, return_value=True)()

    # optimization
    # jax.debug.print('grads = {g}', g=jax.tree.map(jnp.max, grads))
    grads = bst.functional.clip_grad_norm(grads, 1.)
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
        # if (i + 1) % 100 == 0:
        #   self.opt.lr.step_epoch()

        # accuracy
        avg_acc = self.smoother(acc)
        if avg_acc > max_acc:
          max_acc = avg_acc
          if platform.platform().startswith('Linux'):
            self.target.save(i, metrics={'loss': loss, 'acc': acc})
        if max_acc > self.args.acc_th:
          self.print(f'The training accuracy is greater than {self.args.acc_th * 100}%. Training is stopped.')
          break
        t0 = time.time()
    finally:
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
      kappa=8,
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
      X = X.astype(np.float_)

    # can use a greater weight for test period if needed
    return X, match

  def __iter__(self):
    while True:
      yield self.sample_a_trial()


class DecisionMaking(IterableDataset):

  def __init__(
      self, n_in, t_delay=1200
  ):
    super().__init__()

    self.n_in = n_in
    self.t_delay = t_delay
    self.t_interval = 150
    self.f0 = 40. / 1000.
    self.seq_len = int(self.t_interval * 7 + self.t_delay)

  @property
  def num_inputs(self) -> int:
    return self.n_in

  @property
  def num_outputs(self) -> int:
    return 2

  def __iter__(self) -> Iterator[np.ndarray]:
    while True:
      spk_data, _, target_data, _ = self.generate_click_task_data(
        batch_size=1,
        seq_len=self.seq_len,
        n_neuron=self.n_in,
        recall_duration=150,
        prob=0.3,
        t_cue=100,
        n_cues=7,
        t_interval=self.t_interval,
        f0=self.f0,
        n_input_symbols=4
      )
      yield spk_data[0], target_data[0]

  @staticmethod
  @njit
  def generate_click_task_data(batch_size, seq_len, n_neuron, recall_duration, prob, f0=0.5,
                               n_cues=7, t_cue=100, t_interval=150, n_input_symbols=4):
    n_channel = n_neuron // n_input_symbols

    # assign input spike probabilities
    probs = np.where(np.random.random((batch_size, 1)) < 0.5, prob, 1 - prob)

    # for each example in batch, draw which cues are going to be active (left or right)
    cue_assignments = np.asarray(np.random.random(n_cues) > probs, dtype=np.int_)

    # generate input nums - 0: left, 1: right, 2:recall, 3:background noise
    input_nums = 3 * np.ones((batch_size, seq_len), dtype=np.int_)
    input_nums[:, :n_cues] = cue_assignments
    input_nums[:, -1] = 2

    # generate input spikes
    input_spike_prob = np.zeros((batch_size, seq_len, n_neuron))
    d_silence = t_interval - t_cue
    for b in range(batch_size):
      for k in range(n_cues):
        # input channels only fire when they are selected (left or right)
        c = cue_assignments[b, k]
        # reverse order of cues
        i_seq = d_silence + k * t_interval
        i_neu = c * n_channel
        input_spike_prob[b, i_seq:i_seq + t_cue, i_neu:i_neu + n_channel] = f0
    # recall cue
    input_spike_prob[:, -recall_duration:, 2 * n_channel:3 * n_channel] = f0
    # background noise
    input_spike_prob[:, :, 3 * n_channel:] = f0 / 4.
    input_spikes = input_spike_prob > np.random.rand(*input_spike_prob.shape)

    # generate targets
    target_mask = np.zeros((batch_size, seq_len), dtype=np.bool_)
    target_mask[:, -1] = True
    target_nums = (np.sum(cue_assignments, axis=1) > n_cues / 2).astype(np.int_)
    return input_spikes, input_nums, target_nums, target_mask


def get_dms_data(args, cache_dir=os.path.expanduser("./data")):
  _scale = 1.

  try:
    t_fixation = args.t_fixation
  except:
    t_fixation = 500.

  task = DMS(
    dt=bst.environ.get_dt(),
    mode='spiking',
    bg_fr=1.,
    t_fixation=t_fixation * _scale,
    t_sample=500. * _scale,
    t_delay=args.t_delay * _scale,
    t_test=500. * _scale,
    ft_motion=bd.cognitive.Feature(24, 100, 100.)
  )
  train_loader = DataLoader(task, batch_size=args.batch_size)
  test_loader = DataLoader(task, batch_size=args.batch_size)

  in_shape = (task.num_inputs,)
  out_shape = task.num_outputs

  args.warmup_ratio = task.num_steps - task.t_test
  return bst.util.DotDict(
    {'train_loader': train_loader,
     'test_loader': test_loader,
     'in_shape': in_shape,
     'out_shape': out_shape,
     'target_type': 'fixed',  # 'fixed' or 'varied'
     'input_process': lambda x: jnp.asarray(x, dtype=bst.environ.dftype()).transpose(1, 0, 2),
     'label_process': lambda x: jnp.asarray(x, dtype=bst.environ.ditype())}
  )


def get_dm_data(args, cache_dir=None):
  ds = DecisionMaking(n_in=40, t_delay=args.t_delay)
  train_loader = DataLoader(ds, batch_size=args.batch_size)
  test_loader = DataLoader(ds, batch_size=args.batch_size)

  in_shape = (ds.num_inputs,)
  out_shape = ds.num_outputs

  args.warmup_ratio = ds.seq_len - 150
  return bst.util.DotDict(
    {'train_loader': train_loader,
     'test_loader': test_loader,
     'in_shape': in_shape,
     'out_shape': out_shape,
     'target_type': 'fixed',  # 'fixed' or 'varied'
     'input_process': lambda x: jnp.asarray(x, dtype=bst.environ.dftype()).transpose(1, 0, 2),
     'label_process': lambda x: jnp.asarray(x, dtype=bst.environ.ditype())}
  )


def get_long_term_dependent_data(args, cache_dir=os.path.expanduser("./data")):
  data_to_fun = {
    'dms': get_dms_data,
    'dm': get_dm_data,
  }
  ret = data_to_fun[args.dataset.lower()](args, cache_dir)
  return ret


def load_model():
  global global_args
  filepath = 'results/long-term-snn/expsm_diag t_minus_1 exact 0 decay=0.98 dms/optim=adam-lr=0.001-t_delay=1200.0-tau_I2=2000.0-ffscale=10.0-recscale=2.0-2024-06-13 15-39-20'
  filepath = 'results/long-term-snn/expsm_diag t_minus_1 exact 0 decay=0.98 dms/optim=adam-lr=0.001-t_delay=1200.0-tau_I2=2000.0-ffscale=10.0-recscale=2.0-2024-06-13 16-46-58'
  filepath = 'results/long-term-snn/expsm_diag t_minus_1 exact 0 decay=0.98 dms/optim=adam-lr=0.001-t_delay=1200.0-tau_I2=2000.0-ffscale=10.0-recscale=2.0-2024-06-13 16-52-16'
  filepath = 'results/long-term-snn/expsm_diag t_minus_1 exact 0 decay=0.98 dms multi_gaussian/optim=adam-lr=0.001-t_delay=1200.0-tau_I2=2000.0-ffscale=10.0-recscale=2.0-2024-06-13 17-00-44'
  filepath = 'results/long-term-snn/diag t exact 0 dms s2nn/optim=adam-lr=0.001-t_delay=1200.0-tau_I2=2000.0-ffscale=10.0-recscale=2.0-2024-06-13 17-21-55'
  filepath = 'results/long-term-snn/expsm_diag t_minus_1 exact 0 decay=0.985 dms multi_gaussian/optim=adam-lr=0.001-t_delay=2000.0-tau_I2=2000.0-ffscale=10.0-recscale=2.0-2024-06-13 18-19-57'
  with open(f'{filepath}/loss.txt', 'r') as f:
    args = f.readline().strip().replace('Namespace', 'dict')
    global_args = bst.util.DotDict(eval(args))

  def plot_weight_dist(weight_vals: dict, show: bool = True, title='', filepath=None):
    fig, gs = bts.visualize.get_figure(1, len(weight_vals), 3., 4.5)
    for i, (k, v) in enumerate(weight_vals.items()):

      ax = fig.add_subplot(gs[0, i])
      if isinstance(v, dict):
        v = v['weight']
        v1, v2 = jnp.split(v, [n_in], axis=0)
        ax.hist(v1.flatten(), bins=100, density=True, alpha=0.5, label='Input')
        ax.hist(v2.flatten(), bins=100, density=True, alpha=0.5, label='Recurrent')
        plt.legend()
      else:
        ax.hist(v.flatten(), bins=100, density=True, alpha=0.5)
      ax.set_title(k)
    if title:
      plt.suptitle(title)
    if filepath:
      plt.savefig(f'{filepath}/weight_dist-{title}.png')
    if show:
      plt.show()

  bst.environ.set(
    mode=bst.mixin.JointMode(bst.mixin.Batching(), bst.mixin.Training()),
    dt=global_args.dt
  )

  # loading the data
  dataset = get_long_term_dependent_data(global_args)
  global_args.n_out = dataset.out_shape
  n_in = np.prod(dataset.in_shape)

  # creating the network and optimizer
  net = GifNet(
    n_in,
    global_args.n_rec,
    global_args.n_out,
    global_args,
    filepath=filepath
  )
  plot_weight_dist(net.states().subset(bst.ParamState).to_dict_values(), False, 'Before', filepath=filepath)
  net.restore()
  plot_weight_dist(net.states().subset(bst.ParamState).to_dict_values(), False, 'After', filepath=filepath)
  net.verify(dataset.train_loader, dataset.input_process, num_show=5, sps_inc=10., filepath=filepath)


def network_training():
  # environment setting
  bst.environ.set(
    mode=bst.mixin.JointMode(bst.mixin.Batching(), bst.mixin.Training()),
    dt=global_args.dt
  )

  # get file path to output
  if global_args.filepath:
    filepath = global_args.filepath
  else:
    aim = f'long-term-snn/{global_args.exp_name}' if global_args.exp_name else 'long-term-snn'
    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    name = (
      f'optim={global_args.optimizer}-lr={global_args.lr}-t_delay={global_args.t_delay}-'
      f'tau_I2={global_args.tau_I2}-ffscale={global_args.ff_scale}-'
      f'recscale={global_args.rec_scale}-{now}'
    )
    if global_args.method == 'bptt':
      filepath = (
        f'results/{aim}/'
        f'{global_args.method} {global_args.dataset} {global_args.spk_fun}/{name}'
      )
    elif global_args.method == 'diag':
      filepath = (
        f'results/{aim}/{global_args.method} {global_args.vjp_time} '
        f'{global_args.diag_jacobian} {global_args.diag_normalize} '
        f'{global_args.dataset} {global_args.spk_fun}/{name}'
      )
    else:
      filepath = (
        f'results/{aim}/{global_args.method} {global_args.vjp_time} '
        f'{global_args.diag_jacobian} {global_args.diag_normalize} '
        f'decay={global_args.etrace_decay} {global_args.dataset} '
        f'{global_args.spk_fun}/{name}'
      )

  # loading the data
  dataset = get_long_term_dependent_data(global_args)
  global_args.n_out = dataset.out_shape

  # creating the network and optimizer
  net = GifNet(
    np.prod(dataset.in_shape),
    global_args.n_rec,
    global_args.n_out,
    global_args,
    filepath=filepath
  )

  if global_args.mode == 'sim':
    if global_args.filepath:
      net.restore()
    net.verify(dataset.train_loader, dataset.input_process, num_show=5, sps_inc=10.)

  elif global_args.mode == 'train':
    if global_args.optimizer == 'adam':
      opt_cls = bst.optim.Adam
    elif global_args.optimizer == 'momentum':
      opt_cls = bst.optim.Momentum
    elif global_args.optimizer == 'sgd':
      opt_cls = bst.optim.SGD
    else:
      raise ValueError(f'Unknown optimizer: {global_args.optimizer}')
    if global_args.lr_milestones == '':
      lr = global_args.lr
    else:
      lr_milestones = [int(m) for m in global_args.lr_milestones.split(',')]
      lr = bst.optim.MultiStepLR(global_args.lr, milestones=lr_milestones, gamma=0.2)
    opt = opt_cls(lr=lr, weight_decay=global_args.weight_L2)

    # creating the trainer
    trainer = Trainer(
      net,
      opt,
      global_args,
      filepath,
    )
    trainer.f_train(
      dataset.train_loader,
      x_func=dataset.input_process,
      y_func=dataset.label_process,
    )

  else:
    raise ValueError(f'Unknown mode: {global_args.mode}')


if __name__ == '__main__':
  pass
  network_training()
  # load_model()

"""
BPTT:

    python task-rsnn-long-term-dependency.py --mode train --method bptt --dataset dms --t_delay 1200 --n_rec 200 --lr 0.001 --optimizer adam --devices 0


Diag:

    python task-rsnn-long-term-dependency.py --mode train --method diag --diag_jacobian exact --vjp_time t --dataset dms --t_delay 1200 \
        --tau_I2 1500 --n_rec 200 --lr 0.001 --optimizer adam --devices 3

    
    python task-rsnn-long-term-dependency.py --mode train --method diag --diag_jacobian vjp --vjp_time t_minus_1 --dataset dms \
        --t_delay 1500  --tau_I2 2000 --n_rec 200 --lr 0.001 --A2 -0.1 --optimizer adam --devices 1 --t_fixation 10. \
        --spk_fun s2nn --acc_th 0.94

    
    

Expsm_diag:
    
    python task-rsnn-long-term-dependency.py --mode train --method expsm_diag --diag_jacobian vjp --etrace_decay 0.98 --vjp_time t_minus_1 \
        --dataset dms --t_delay 1500  --tau_I2 2000 --n_rec 200 --lr 0.001 --A2 -0.1 --optimizer adam --devices 1 --t_fixation 10. \
        --spk_fun s2nn --acc_th 0.94

    python task-rsnn-long-term-dependency.py --mode train --method expsm_diag --diag_jacobian vjp --etrace_decay 0.98 --vjp_time t_minus_1 \
        --dataset dms --t_delay 1800  --tau_I2 2000 --n_rec 200 --lr 0.001 --A2 -0.1 --optimizer adam --devices 3 --t_fixation 10. \
        --spk_fun multi_gaussian --acc_th 0.94

"""

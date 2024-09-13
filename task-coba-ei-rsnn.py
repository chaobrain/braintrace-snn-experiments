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

import os
import os.path
import pickle
import platform
import time
from typing import Callable
from typing import Tuple

import matplotlib

from utils import MyArgumentParser

if platform.system().startswith('Linux'):
  matplotlib.use('agg')

parser = MyArgumentParser()

# Learning parameters
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs.")

# Dataset
parser.add_argument("--task", type=str, default='ea', choices=['ea', 'dms'], help="")
parser.add_argument("--batch_size", type=int, default=128, help="")
parser.add_argument("--warmup", type=float, default=0., help="The ratio for network simulation.")
parser.add_argument("--num_workers", type=int, default=4, help="")

# Model
parser.add_argument("--diff_spike", type=int, default=0, help="0: False, 1: True")
parser.add_argument("--dt", type=float, default=1., help="")
parser.add_argument("--net", type=str, default='coba', choices=['coba', 'cuba'], help="")
parser.add_argument("--n_rec", type=int, default=200, help="")
parser.add_argument("--w_ei_ratio", type=float, default=4., help="")
parser.add_argument("--ff_scale", type=float, default=1., help="")
parser.add_argument("--rec_scale", type=float, default=0.5, help="")
parser.add_argument("--beta", type=float, default=1.0, help="")
parser.add_argument("--tau_a", type=float, default=1000., help="")
parser.add_argument("--tau_neu", type=float, default=100., help="")
parser.add_argument("--tau_syn", type=float, default=10., help="")
parser.add_argument("--tau_out", type=float, default=10., help="")
parser.add_argument("--exp_name", type=str, default='', help="")

# Training parameters
parser.add_argument("--mode", type=str, default='train', choices=['sim', 'train'], help="")

# Regularization parameters
parser.add_argument("--weight_L1", type=float, default=0.0, help="The weight L1 regularization.")
parser.add_argument("--weight_L2", type=float, default=0.0, help="The weight L2 regularization.")
gargs = parser.parse_args()

import brainstate as bst
import brainpy as bp
import brainpy_datasets as bd
import brainscale as bnn
import braintools as bts
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, IterableDataset

bst.environ.set(
  dt=gargs.dt,
  mode=bst.mixin.JointMode(bst.mixin.Training(), bst.mixin.Batching())
)
diag_norm_mapping = {
  0: None,
  1: True,
  2: False
}


class TaskData(IterableDataset):
  def __init__(self, task: bd.cognitive.CognitiveTask):
    self.task = task

  def __iter__(self):
    while True:
      yield self.task.sample_a_trial(0)[:2]


class TaskLoader(DataLoader):
  def __init__(self, dataset: bd.cognitive.CognitiveTask, *args, **kwargs):
    assert isinstance(dataset, bd.cognitive.CognitiveTask)
    super().__init__(TaskData(dataset), *args, **kwargs)


class GIF(bst.nn.Neuron):
  def __init__(
      self, size, V_rest=0., V_th_inf=1., tau=20., tau_a=50., beta=1.,
      V_initializer: Callable = bst.init.Uniform(0., 1.),
      I2_initializer: Callable = bst.init.ZeroInit(),
      spike_fun: Callable = bst.surrogate.ReluGrad(),
      spk_reset: str = 'soft',
      keep_size: bool = False,
      name: str = None,
      mode: bst.mixin.Mode = None,
  ):
    super().__init__(size, keep_size=keep_size, name=name, mode=mode, spk_fun=spike_fun, spk_reset=spk_reset)

    # params
    self.V_rest = bst.init.param(V_rest, self.varshape, allow_none=False)
    self.V_th_inf = bst.init.param(V_th_inf, self.varshape, allow_none=False)
    self.tau = bst.init.param(tau, self.varshape, allow_none=False)
    self.tau_I2 = bst.init.param(tau_a, self.varshape, allow_none=False)
    self.beta = bst.init.param(beta, self.varshape, allow_none=False)

    # initializers
    self._V_initializer = V_initializer
    self._I_initializer = I2_initializer

  def init_state(self, batch_size=None):
    self.V = bnn.ETraceVar(bst.init.param(self._V_initializer, self.varshape, batch_size))
    self.I_adp = bnn.ETraceVar(bst.init.param(self._I_initializer, self.varshape, batch_size))

  def dI2(self, I2, t):
    return - I2 / self.tau_I2

  def dV(self, V, t, I_ext):
    I_ext = self.sum_current_inputs(V, init=I_ext)
    return (- V + self.V_rest + I_ext) / self.tau

  def update(self, x=0.):
    t = bst.environ.get('t')
    last_spk = self.get_spike()
    if gargs.diff_spike == 0:
      last_spk = jax.lax.stop_gradient(last_spk)
    last_V = self.V.value - self.V_th_inf * last_spk
    last_I2 = self.I_adp.value - self.beta * last_spk

    I2 = bst.nn.exp_euler_step(self.dI2, last_I2, t)
    V = bst.nn.exp_euler_step(self.dV, last_V, t, I_ext=(x + I2))
    V += self.sum_delta_inputs()
    self.I_adp.value = I2
    self.V.value = V

    # outputs
    mem = (V - self.V_th_inf) / self.V_th_inf
    mem = jax.nn.standardize(mem, axis=-1)
    return mem

  def get_spike(self, V=None):
    V = self.V.value if V is None else V
    spk = self.spk_fun((V - self.V_th_inf) / self.V_th_inf)
    return spk


class _SNNEINet(bst.Module):
  def __init__(
      self, n_in, n_rec, n_out, tau_neu=10., tau_a=100., beta=1., tau_syn=10., tau_out=10.,
      ff_scale=1., rec_scale=1., E_exc=None, E_inh=None, w_ei_ratio: float = 10., filepath=None,
  ):
    super().__init__()

    self.filepath = filepath
    self.n_exc = int(n_rec * 0.8)
    self.n_inh = n_rec - self.n_exc

    # neurons
    tau_a = bst.random.uniform(100., tau_a * 2., n_rec)
    self.pop = GIF(n_rec, tau=tau_neu, tau_a=tau_a, beta=beta)
    ff_init = bst.init.KaimingNormal(scale=ff_scale)
    # feedforward
    self.ff2r = bst.nn.HalfProjAlignPostMg(
      comm=bnn.SignedWLinear(n_in, n_rec, w_init=ff_init),
      syn=bst.nn.Expon.delayed(size=n_rec, tau=tau_syn),
      out=(bst.nn.CUBA.delayed() if E_exc is None else bst.nn.COBA.delayed(E=E_exc)),
      post=self.pop
    )
    # recurrent
    inh_init = bst.init.KaimingNormal(scale=rec_scale * w_ei_ratio)
    exc_init = bst.init.KaimingNormal(scale=rec_scale)
    inh2r_conn = bnn.SignedWLinear(self.n_inh, n_rec, w_init=inh_init, w_sign=-1. if E_inh is None else None)
    exc2r_conn = bnn.SignedWLinear(self.n_exc, n_rec, w_init=exc_init)

    self.inh2r = bst.nn.HalfProjAlignPostMg(
      comm=inh2r_conn,
      syn=bnn.Expon.delayed(size=n_rec, tau=tau_syn),
      out=(bst.nn.CUBA.delayed() if E_inh is None else bst.nn.COBA.delayed(E=E_inh)),
      post=self.pop
    )
    self.exc2r = bst.nn.HalfProjAlignPostMg(
      comm=exc2r_conn,
      syn=bnn.Expon.delayed(size=n_rec, tau=tau_syn),
      out=(bst.nn.CUBA.delayed() if E_exc is None else bst.nn.COBA.delayed(E=E_exc)),
      post=self.pop
    )
    # output
    self.out = bnn.LeakyRateReadout(n_rec, n_out, tau=tau_out)

  def update(self, spk):
    e_sps, i_sps = jnp.split(self.pop.get_spike(), [self.n_exc], axis=-1)
    self.ff2r(spk)
    self.exc2r(e_sps)
    self.inh2r(i_sps)
    return self.out(self.pop())

  def save_state(self, **kwargs):
    states = {
      'pop.tau_I2': self.pop.tau_I2,
      'ff2r.weight': self.ff2r.comm.weight_op.value,
      'exc2r.weight': self.exc2r.comm.weight_op.value,
      'inh2r.weight': self.inh2r.comm.weight_op.value,
      'out.weight': self.out.weight_op.value
    }
    states = jax.tree.map(np.asarray, states)
    if self.filepath is not None:
      with open(f'{self.filepath}/states.pkl', 'wb') as f:
        pickle.dump(states, f)

  def load_state(self):
    if self.filepath is None:
      return
    with open(f'{self.filepath}/states.pkl', 'rb') as f:
      state_dict = pickle.load(f)
    self.ff2r.comm.weight_op.value = state_dict['ff2r.weight']
    self.exc2r.comm.weight_op.value = state_dict['exc2r.weight']
    self.inh2r.comm.weight_op.value = state_dict['inh2r.weight']
    self.out.weight_op.value = state_dict['out.weight']
    self.pop.tau_I2 = state_dict['pop.tau_I2']

  def vis_data(self):
    n_rec = self.pop.num
    return {
      'rec_spk': self.pop.get_spike(),
      'rec_mem': self.pop.V.value[:, np.arange(0, n_rec, n_rec // 10)],
      'out': self.out.r.value,
    }

  @bst.transform.jit(static_argnums=0)
  def predict(self, batched_inputs):
    def step(i, inp):
      with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
        self.update(inp)
      spk = self.pop.get_spike()
      rec_mem = self.pop.V.value
      out = self.out.r.value
      return spk, rec_mem, out

    n_seq = batched_inputs.shape[0]
    batch_size = batched_inputs.shape[1]

    bst.init_states(self, batch_size)
    indices = np.arange(n_seq)
    res = bst.transform.for_loop(step, indices, batched_inputs, pbar=bst.transform.ProgressBar(10))
    return res

  def visualize(self, inputs, n2show: int = 5, filename: str = None):
    n_seq = inputs.shape[0]
    indices = np.arange(n_seq)
    batch_size = inputs.shape[1]
    bst.init_states(self, batch_size)

    def step(i, inp):
      with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
        self.update(inp)
      return self.vis_data()

    res = bst.transform.for_loop(step, indices, inputs, pbar=bst.transform.ProgressBar(10))

    fig, gs = bp.visualize.get_figure(4, n2show, 3., 4.5)
    for i in range(n2show):
      # input spikes
      bp.visualize.raster_plot(indices, inputs[:, i], ax=fig.add_subplot(gs[0, i]), xlim=(0, n_seq))
      # recurrent spikes
      bp.visualize.raster_plot(indices, res['rec_spk'][:, i], ax=fig.add_subplot(gs[1, i]), xlim=(0, n_seq))
      # recurrent membrane potentials
      ax = fig.add_subplot(gs[2, i])
      ax.plot(indices, res['rec_mem'][:, i])
      # output potentials
      ax = fig.add_subplot(gs[3, i])
      ax.plot(indices, res['out'][:, i])

    if filename is None:
      plt.show()
      plt.close()
    else:
      plt.savefig(filename)
      plt.close()


class SNNCubaNet(_SNNEINet):
  def __init__(
      self, n_in, n_rec, n_out, tau_neu=10., tau_a=100., beta=1., tau_syn=10., tau_out=10.,
      ff_scale=1., rec_scale=1., w_ei_ratio=4., filepath=None
  ):
    super().__init__(
      n_in=n_in,
      n_rec=n_rec,
      n_out=n_out,
      tau_neu=tau_neu,
      tau_a=tau_a, beta=beta,
      tau_syn=tau_syn,
      tau_out=tau_out,
      ff_scale=ff_scale,
      rec_scale=rec_scale,
      E_exc=None,
      E_inh=None,
      w_ei_ratio=w_ei_ratio,
      filepath=filepath,
    )


class SNNCobaNet(_SNNEINet):
  def __init__(
      self, n_in, n_rec, n_out, tau_neu=10.,
      tau_a=100., beta=1., tau_syn=10., tau_out=10.,
      ff_scale=1., rec_scale=1., w_ei_ratio=4.,
      filepath=None,
  ):
    super().__init__(
      n_in=n_in,
      n_rec=n_rec,
      n_out=n_out,
      tau_neu=tau_neu,
      tau_a=tau_a,
      beta=beta,
      tau_syn=tau_syn,
      tau_out=tau_out,
      ff_scale=ff_scale,
      rec_scale=rec_scale,
      E_exc=5.,
      E_inh=-10.,
      w_ei_ratio=w_ei_ratio,
      filepath=filepath,
    )


class Trainer:
  def __init__(
      self,
      target_net: _SNNEINet,
      optimizer: bst.optim.Optimizer,
      loader: bd.cognitive.TaskLoader,
      args: bst.util.DotDict,
      filepath: str | None = None
  ):
    # the network
    self.target = target_net

    # the dataset
    self.loader = loader

    # parameters
    self.args = args
    self.filepath = filepath

    # optimizer
    weights = self.target.states().subset(bst.ParamState)
    self.optimizer = optimizer
    self.optimizer.register_trainable_weights(weights)

  def print(self, msg, file=None):
    if file is not None:
      print(msg, file=file)
    print(msg)

  def _loss(self, out, target):
    # MSE loss
    mse_loss = bts.metric.softmax_cross_entropy_with_integer_labels(out, target).mean()
    # L1 regularization loss
    l1_loss = 0.
    if self.args.weight_L1 != 0.:
      leaves = self.target.states().subset(bst.ParamState).to_dict_values()
      for leaf in leaves.values():
        l1_loss += self.args.weight_L1 * jnp.sum(jnp.abs(leaf))
    return mse_loss, l1_loss

  def _acc(self, outs, target):
    pred = jnp.argmax(jnp.sum(outs, 0), 1)  # [T, B, N] -> [B, N] -> [B]
    # pred = jnp.argmax(jnp.max(outs, 0), 1)  # [T, B, N] -> [B, N] -> [B]
    acc = jnp.asarray(pred == target, dtype=bst.environ.dftype()).mean()
    return acc

  @bst.transform.jit(static_argnums=(0,))
  def etrace_train(self, inputs, targets):
    inputs = jnp.asarray(inputs, dtype=bst.environ.dftype())
    indices = np.arange(inputs.shape[0])
    bst.init_states(self.target, inputs.shape[1])
    weights = self.target.states().subset(bst.ParamState)
    warmup = self.args.warmup + inputs.shape[0] if self.args.warmup < 0 else self.args.warmup
    n_sim = int(warmup) if warmup > 0 else 0

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
        diag_normalize=diag_norm_mapping[self.args.diag_normalize],
        vjp_time=self.args.vjp_time,
      )
      model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
    elif self.args.method == 'diag':
      model = bnn.DiagParamDimAlgorithm(
        _single_step,
        diag_normalize=diag_norm_mapping[self.args.diag_normalize],
        vjp_time=self.args.vjp_time,
      )
      model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
    elif self.args.method == 'hybrid':
      model = bnn.DiagHybridDimAlgorithm(
        _single_step,
        self.args.etrace_decay,
        diag_normalize=diag_norm_mapping[self.args.diag_normalize],
        vjp_time=self.args.vjp_time,
      )
      model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
    else:
      raise ValueError(f'Unknown online learning methods: {self.args.method}.')

    model.graph.show_graph()

    def _etrace_grad(i, inp):
      # call the model
      out = model(i, inp, running_index=i)
      # calculate the loss
      me, l1 = self._loss(out, targets)
      return me + l1, (out, me, l1)

    def _etrace_step(prev_grads, x):
      # no need to return weights and states, since they are generated then no longer needed
      i, inp = x
      f_grad = bst.transform.grad(_etrace_grad, grad_vars=weights, has_aux=True, return_value=True)
      cur_grads, local_loss, (out, mse_l, l1) = f_grad(i, inp)
      next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
      return next_grads, (out, mse_l, l1)

    def _etrace_train(indices_, inputs_):
      # forward propagation
      grads = jax.tree.map(jnp.zeros_like, weights.to_dict_values())
      grads, (outs, mse_ls, l1_ls) = bst.transform.scan(_etrace_step, grads, (indices_, inputs_))
      acc = self._acc(outs, targets)

      grads = bst.functional.clip_grad_norm(grads, 1.)
      self.optimizer.update(grads)
      # accuracy
      return mse_ls.mean(), l1_ls.mean(), acc

    # running indices
    if n_sim > 0:
      bst.transform.for_loop(lambda i, inp: model(i, inp, running_index=i), indices[:n_sim], inputs[:n_sim])
      r = _etrace_train(indices[n_sim:], inputs[n_sim:])
    else:
      r = _etrace_train(indices, inputs)
    return r

  @bst.transform.jit(static_argnums=(0,))
  def bptt_train(self, inputs, targets) -> Tuple:
    inputs = jnp.asarray(inputs, dtype=bst.environ.dftype())
    indices = jnp.arange(inputs.shape[0])
    bst.init_states(self.target, inputs.shape[1])
    weights = self.target.states().subset(bst.ParamState)
    warmup = self.args.warmup + inputs.shape[0] if self.args.warmup < 0 else self.args.warmup
    n_sim = int(warmup) if warmup > 0 else 0

    def _step_run(i, inp):
      with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
        out = self.target(inp)
      return self._loss(out, targets), out

    def _bptt_grad():
      (mse_losses, l1_losses), outs = bst.transform.for_loop(_step_run, indices, inputs)
      mse_losses = mse_losses[n_sim:].mean()
      l1_losses = l1_losses[n_sim:].mean()
      acc = self._acc(outs[n_sim:], targets)
      return mse_losses + l1_losses, (mse_losses, l1_losses, acc)

    f_grad = bst.transform.grad(_bptt_grad, grad_vars=weights, has_aux=True, return_value=True)
    grads, loss, (mse_losses, l1_losses, acc) = f_grad()
    grads = bst.functional.clip_grad_norm(grads, 1.)
    self.optimizer.update(grads)
    return mse_losses, l1_losses, acc

  def f_sim(self):
    inputs, outputs = next(iter(self.loader))
    inputs = jnp.asarray(inputs, dtype=bst.environ.dftype()).transpose(1, 0, 2)
    self.target.visualize(inputs)

  def f_train(self):
    file = None
    if self.filepath is not None:
      if not os.path.exists(self.filepath):
        os.makedirs(self.filepath)
      file = open(f'{self.filepath}/loss.txt', 'w')
    self.print(self.args, file=file)

    acc_max = 0.
    t0 = time.time()
    for bar_idx, (inputs, outputs) in enumerate(self.loader):
      if bar_idx > gargs.epochs:
        break

      inputs = jnp.asarray(inputs, dtype=bst.environ.dftype()).transpose(1, 0, 2)
      outputs = jnp.asarray(outputs, dtype=bst.environ.ditype())
      mse_ls, l1_ls, acc = (self.bptt_train(inputs, outputs)
                            if self.args.method == 'bptt' else
                            self.etrace_train(inputs, outputs))
      if (bar_idx + 1) % 100 == 0:
        self.optimizer.lr.step_epoch()
      desc = (f'Batch {bar_idx:2d}, '
              f'CE={float(mse_ls):.8f}, '
              f'L1={float(l1_ls):.6f}, '
              f'acc={float(acc):.6f}, '
              f'time={time.time() - t0:.2f} s')
      self.print(desc, file=file)

      if acc > acc_max:
        acc_max = acc
        if self.filepath is not None:
          self.target.save_state()
          self.target.visualize(inputs, filename=f'{self.filepath}/train-results-{bar_idx}.png')

      t0 = time.time()
      if acc_max > 0.95:
        print(f'Accuracy reaches 95% at {bar_idx}th epoch. Stop training.')
        break
    if file is not None:
      file.close()


def training():
  # filepath
  now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
  exp_name = 'results/ei-coba-rsnn/'
  if gargs.exp_name:
    exp_name += gargs.exp_name + '/'
  if gargs.method == 'bptt':
    filepath = f'{exp_name}/{gargs.method}/tau_a={gargs.tau_a}-tau_neu={gargs.tau_neu}-tau_syn={gargs.tau_syn}-{now}'
  else:
    filepath = f'{exp_name}/{gargs.method}-{gargs.diag_normalize}/tau_a={gargs.tau_a}-tau_neu={gargs.tau_neu}-tau_syn={gargs.tau_syn}-{now}'

  # data
  task = bd.cognitive.EvidenceAccumulation(dt=bst.environ.get_dt(), mode='spiking', )
  gargs.warmup = -(task.t_recall / bst.environ.get_dt())
  loader = TaskLoader(task, batch_size=gargs.batch_size, num_workers=gargs.num_workers)

  # network
  cls = SNNCobaNet if gargs.net == 'coba' else SNNCubaNet
  net = cls(
    task.num_inputs,
    gargs.n_rec,
    task.num_outputs,
    beta=gargs.beta,
    tau_a=gargs.tau_a,
    tau_neu=gargs.tau_neu,
    tau_syn=gargs.tau_syn,
    tau_out=gargs.tau_out,
    ff_scale=gargs.ff_scale,
    rec_scale=gargs.rec_scale,
    w_ei_ratio=gargs.w_ei_ratio,
    filepath=filepath,
  )

  # optimizer
  opt = bst.optim.Adam(lr=gargs.lr)

  # trainer
  trainer = Trainer(net, opt, loader, gargs, filepath=filepath)
  if gargs.mode == 'sim':
    trainer.f_sim()
  else:
    trainer.f_train()


def load_model():
  plt.style.use(['science', 'nature', 'notebook'])

  different_path = {
    'BPTT': 'results/ei-coba-rsnn/diff-spike/bptt/tau_a=1500.0-tau_neu=100.0-tau_syn=5.0-2024-07-05 16-06-03',
    'D-RTRL': 'results/ei-coba-rsnn/diff-spike/diag-0/tau_a=1500.0-tau_neu=400.0-tau_syn=5.0-2024-07-05 16-05-14',
    'ES-D-RTRL': 'results/ei-coba-rsnn/diff-spike/expsm_diag-0/tau_a=1500.0-tau_neu=400.0-tau_syn=5.0-2024-07-05 16-05-45',
  }
  category = 'diff'

  # different_path = {
  #   'BPTT': 'results/ei-coba-rsnn/non-diff-spike/bptt/tau_a=1500.0-tau_neu=100.0-tau_syn=5.0-2024-07-05 19-03-27',
  #   'D-RTRL': 'results/ei-coba-rsnn/non-diff-spike/diag-0/tau_a=1500.0-tau_neu=400.0-tau_syn=5.0-2024-07-05 19-11-44',
  #   'ES-D-RTRL': 'results/ei-coba-rsnn/non-diff-spike/expsm_diag-0/tau_a=1500.0-tau_neu=400.0-tau_syn=5.0-2024-07-05 16-35-29',
  # }
  # category = 'non-diff'

  def _plot_weight_dist(inh_weight, exc_weight, title=''):
    fig, gs = bts.visualize.get_figure(1, 2, 4.5, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(np.abs(inh_weight).flatten(), bins=100, alpha=0.5)
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(np.abs(exc_weight).flatten(), bins=100, alpha=0.5)
    if title:
      plt.suptitle(title)
    plt.show()

  def _plot_recurrent_spikes(spikes, savepath=None):
    spikes = np.squeeze(spikes)
    times = np.arange(spikes.shape[0])
    elements = np.where(spikes > 0.)
    times, index = times[elements[0]], elements[1]

    fig, gs = bts.visualize.get_figure(1, 1, 4.5, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(times, index, 'k.', markersize=2)
    plt.xlim(0, spikes.shape[0])
    n_exc = gargs.n_rec * 3 / 4
    plt.axhline(y=n_exc, color='k', linestyle='--', linewidth=0.5)
    plt.yticks(
      [n_exc // 2, (gargs.n_rec - n_exc) // 2 + n_exc],
      ['Excitatory', 'Inhibitory'],
      rotation=90,
      fontsize=14,
      va='center'
    )
    plt.xlabel('Time [ms]')
    plt.title('Recurrent Spiking Activity')
    if savepath is not None:
      plt.savefig(savepath)
    else:
      plt.show()

  def _plot_membrane_potentials(mem, savepath=None):
    n2vis = 4
    mem = np.squeeze(mem)
    n_exc = int(gargs.n_rec * 3 / 4)
    times = np.arange(mem.shape[0])
    axes = []

    fig, gs = bts.visualize.get_figure(2, 1, 4.5 // 2, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    axes.append(ax)
    for i in range(0, n_exc, n_exc // n2vis):
      ax.plot(times, mem[:, i])
    ax.set_ylabel('Excitatory')
    ax.set_xticks([])
    ax.set_xlim(0, mem.shape[0])
    plt.title('Recurrent Membrane Potential')

    ax = fig.add_subplot(gs[1, 0])
    axes.append(ax)
    for i in range(n_exc, gargs.n_rec, (gargs.n_rec - n_exc) // n2vis):
      ax.plot(times, mem[:, i])
    ax.set_ylabel('Inhibitory')
    ax.set_xlabel('Time [ms]')
    ax.set_xlim(0, mem.shape[0])

    fig.align_ylabels(axes)
    if savepath is not None:
      plt.savefig(savepath)
    else:
      plt.show()

  def _plot_sorted_spikes(spikes, title='', savepath=None):
    """
    Plots the spikes sorted by the first spike time of each neuron.

    Parameters:
    - spikes: A numpy array of shape (time, neurons) where each element is a boolean indicating a spike.
    """
    spikes = np.squeeze(spikes)
    spike_times, neuron_indices = np.where(spikes > 0)

    # Determine the first spike time for each neuron
    first_spike_times = np.array(
      [spike_times[neuron_indices == i].min()
       if np.any(neuron_indices == i) else np.inf
       for i in range(spikes.shape[1])]
    )

    # Sort neurons by their first spike time
    sorted_neurons = np.argsort(first_spike_times)

    fig, gs = bts.visualize.get_figure(1, 1, 4.5, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    # Plot each spike, adjusting the neuron index based on the sorted order
    final_exc_times, final_exc_indices = [], []
    final_inh_times, final_inh_indices = [], []
    for time_, neuron in zip(spike_times, neuron_indices):
      sorted_index = np.where(sorted_neurons == neuron)[0][0]
      if neuron > 300:
        final_exc_times.append(time_)
        final_exc_indices.append(sorted_index)
        ax.plot(time_, sorted_index, 'm.', markersize=2)
      else:
        final_inh_times.append(time_)
        final_inh_indices.append(sorted_index)
        ax.plot(time_, sorted_index, 'y.', markersize=2)
    # ax.plot(final_exc_times, final_exc_indices, 'm.', markersize=2, label='Excitatory')
    # ax.plot(final_inh_times, final_inh_indices, 'y.', markersize=2, label='Inhibitory')
    # ax.legend(fontsize=12, fancybox=True, framealpha=0.5,  bbox_to_anchor=(1.0, 1.0))
    # Adjusting the plot
    plt.xlim(0, spikes.shape[0])
    plt.xlabel('Time [ms]')
    plt.ylabel('Neuron Index (sorted)')
    plt.yticks([])
    if title:
      plt.title(title)
    if savepath is not None:
      plt.savefig(savepath)
    else:
      plt.show()
    plt.close()

  def _plot_spikes(spikes, savepath=None):
    spikes = np.squeeze(spikes)
    times = np.arange(spikes.shape[0])
    elements = np.where(spikes > 0.)
    times, index = times[elements[0]], elements[1]
    sort_idx = np.argsort(index)
    times, index = times[sort_idx], index[sort_idx]

    fig, gs = bts.visualize.get_figure(1, 1, 4.5, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(times, index, 'k.', markersize=2)
    if savepath is not None:
      plt.savefig(savepath)
    else:
      plt.show()

  task = bd.cognitive.EvidenceAccumulation(dt=bst.environ.get_dt(), mode='spiking', )

  global gargs
  for key, filepath in different_path.items():
    with open(f'{filepath}/loss.txt', 'r') as f:
      print(f'Loading {filepath} ...')

      # parameters
      args = f.readline().strip().replace('Namespace', 'dict')
      gargs = bst.util.DotDict(eval(args))

      # environment
      bst.environ.set(
        mode=bst.mixin.JointMode(bst.mixin.Batching(), bst.mixin.Training()),
        dt=gargs.dt
      )
      bst.util.clear_name_cache()

      # task
      gargs.warmup = -(task.t_recall / bst.environ.get_dt())
      xs = np.expand_dims(task.sample_a_trial(0)[0], axis=1)

      # networks
      net = SNNCobaNet(
        task.num_inputs,
        gargs.n_rec,
        task.num_outputs,
        beta=gargs.beta,
        tau_a=gargs.tau_a,
        tau_neu=gargs.tau_neu,
        tau_syn=gargs.tau_syn,
        tau_out=gargs.tau_out,
        ff_scale=gargs.ff_scale,
        rec_scale=gargs.rec_scale,
        w_ei_ratio=gargs.w_ei_ratio,
        filepath=filepath,
      )
      net.load_state()

      # inh_w = net.inh2r.comm.weight_op.value
      # exc_w = net.exc2r.comm.weight_op.value
      # _plot_weight_dist(inh_w, exc_w, title=key)

      # spks, mems, outs = net.predict(xs)

      # _plot_spikes(spks)
      # _plot_sorted_spikes(spks)
      # _plot_sorted_spikes(spks, savepath=f'results/ei-coba-rsnn/{category}-{key}-sorted-spikes.eps',
      #                     # title='With Surrogate Gradient' if category == 'diff' else 'Without Surrogate Gradient'
      #                     title=key
      #                     )

      # _plot_recurrent_spikes(spks, savepath=f'results/ei-coba-rsnn/{category}-{key}-rec-spikes.eps'
      #                        )
      # _plot_membrane_potentials(mems, savepath=f'results/ei-coba-rsnn/{category}-{key}-mem-potentials.eps'
      #                           )


def compare_firing_rate_and_others():
  plt.style.use(['science', 'nature', 'notebook'])

  different_path = {
    'BPTT': 'results/ei-coba-rsnn/diff-spike/bptt/tau_a=1500.0-tau_neu=100.0-tau_syn=5.0-2024-07-05 16-06-03',
    'D-RTRL': 'results/ei-coba-rsnn/diff-spike/diag-0/tau_a=1500.0-tau_neu=400.0-tau_syn=5.0-2024-07-05 16-05-14',
    'ES-D-RTRL': 'results/ei-coba-rsnn/diff-spike/expsm_diag-0/tau_a=1500.0-tau_neu=400.0-tau_syn=5.0-2024-07-05 16-05-45',
  }

  def get_ei_spiking_ratio(filepath):
    global gargs
    with open(f'{filepath}/loss.txt', 'r') as f:
      print(f'Loading {filepath} ...')
      args = f.readline().strip().replace('Namespace', 'dict')
      gargs = bst.util.DotDict(eval(args))
      bst.environ.set(mode=bst.mixin.JointMode(bst.mixin.Batching(), bst.mixin.Training()), dt=gargs.dt)
      bst.util.clear_name_cache()
      gargs.warmup = -(task.t_recall / bst.environ.get_dt())
      net = SNNCobaNet(
        task.num_inputs,
        gargs.n_rec,
        task.num_outputs,
        beta=gargs.beta,
        tau_a=gargs.tau_a,
        tau_neu=gargs.tau_neu,
        tau_syn=gargs.tau_syn,
        tau_out=gargs.tau_out,
        ff_scale=gargs.ff_scale,
        rec_scale=gargs.rec_scale,
        w_ei_ratio=gargs.w_ei_ratio,
        filepath=filepath,
      )
      net.load_state()

      ei_ratios, fr = [], []
      for _ in range(10):
        xs = np.expand_dims(task.sample_a_trial(0)[0], axis=1)
        spks, mems, outs = net.predict(xs)
        ei_ratios.append(np.sum(spks[1500:, 0, :300]) / (np.sum(spks[1500:, 0])))
        fr.append(np.sum(spks) / (spks.shape[0] * spks.shape[1]) / spks.shape[2] * 1000)
    return np.asarray(ei_ratios), np.asarray(fr)

  task = bd.cognitive.EvidenceAccumulation(dt=bst.environ.get_dt(), mode='spiking', )
  ei_ratio1, firing_rates1 = get_ei_spiking_ratio(different_path['BPTT'])
  print('BPTT:', ei_ratio1)
  ei_ratio2, firing_rates2 = get_ei_spiking_ratio(different_path['D-RTRL'])
  print('D-RTRL:', ei_ratio2)
  ei_ratio3, firing_rates3 = get_ei_spiking_ratio(different_path['ES-D-RTRL'])
  print('ES-D-RTRL:', ei_ratio3)

  fig, gs = bts.visualize.get_figure(1, 1, 3.0, 4.5)
  ax = fig.add_subplot(gs[0, 0])
  ax.violinplot([ei_ratio1, ei_ratio2, ei_ratio3], showmeans=False, showmedians=True)
  # ax.boxplot([ei_ratio1, ei_ratio2, ei_ratio3])
  ax.set_ylabel('E/I Spiking Ratio')
  plt.grid(True)
  ax.set_xticks([y + 1 for y in range(3)], labels=['BPTT', 'D-RTRL', 'ES-D-RTRL'])
  # plt.show()

  fig, gs = bts.visualize.get_figure(1, 1, 3.0, 4.5)
  ax = fig.add_subplot(gs[0, 0])
  # ax.violinplot([firing_rates1, firing_rates2, firing_rates3], showmeans=False, showmedians=True)
  ax.boxplot([firing_rates1, firing_rates2, firing_rates3])
  ax.set_ylabel('Firing Rate [Hz]')
  plt.grid(True)
  ax.set_xticks([y + 1 for y in range(3)], labels=['BPTT', 'D-RTRL', 'ES-D-RTRL'])
  plt.show()


if __name__ == '__main__':
  pass
  # training()
  # load_model()
  compare_firing_rate_and_others()
  # compare_d_rtrl()

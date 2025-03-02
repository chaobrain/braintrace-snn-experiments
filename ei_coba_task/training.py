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
import sys
import time
from typing import Tuple

# Linux
sys.path.append('/mnt/d/codes/projects/brainscale')
sys.path.append('/mnt/d/codes/projects/brainstate')
sys.path.append('/mnt/d/codes/projects/brainevent')

# windows
sys.path.append('D:/codes/projects/brainscale')
sys.path.append('D:/codes/projects/brainstate')
sys.path.append('D:/codes/projects/brainevent')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from args import parse_args
import matplotlib
matplotlib.use('agg')

import brainstate
import braintools
import brainscale
import brainunit as u
import jax
import jax.numpy as jnp

from data import EvidenceAccumulation
from model import _SNNEINet, SNNCubaNet, SNNCobaNet


class Trainer:
    def __init__(
        self,
        target_net: _SNNEINet,
        optimizer: brainstate.optim.Optimizer,
        loader: EvidenceAccumulation,
        args: brainstate.util.DotDict,
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
        self.trainable_weights = self.target.states().subset(brainstate.ParamState)
        self.optimizer = optimizer
        self.optimizer.register_trainable_weights(self.trainable_weights)

    def print(self, msg, file=None):
        if file is not None:
            print(msg, file=file)
        print(msg)

    def _loss(self, out, target):
        # MSE loss
        mse_loss = braintools.metric.softmax_cross_entropy_with_integer_labels(out, target).mean()
        # L1 regularization loss
        l1_loss = 0.
        if self.args.weight_L1 != 0.:
            leaves = self.trainable_weights.to_dict_values()
            for leaf in leaves.values():
                l1_loss += self.args.weight_L1 * jnp.sum(jnp.abs(leaf))
        return mse_loss, l1_loss

    def _acc(self, outs, target):
        pred = jnp.argmax(jnp.sum(outs, 0), 1)  # [T, B, N] -> [B, N] -> [B]
        acc = jnp.asarray(pred == target, dtype=brainstate.environ.dftype()).mean()
        return acc

    @brainstate.compile.jit(static_argnums=(0,))
    def etrace_train(self, inputs, targets):
        inputs = jnp.asarray(inputs, dtype=brainstate.environ.dftype())
        # inputs: [n_seq, n_batch, n_feat]
        n_batch = inputs.shape[1]

        # initialize the online learning model
        if self.args.method == 'esd-rtrl':
            model = brainscale.IODimVjpAlgorithm(
                self.target,
                self.args.etrace_decay,
                vjp_method=self.args.vjp_method,
            )
        elif self.args.method == 'd-rtrl':
            model = brainscale.ParamDimVjpAlgorithm(
                self.target,
                vjp_method=self.args.vjp_method,
            )
        elif self.args.method == 'hybrid':
            model = brainscale.HybridDimVjpAlgorithm(
                self.target,
                self.args.etrace_decay,
                vjp_method=self.args.vjp_method,
            )
        else:
            raise ValueError(f'Unknown online learning methods: {self.args.method}.')

        @brainstate.augment.vmap_new_states(state_tag='new', axis_size=n_batch)
        def init():
            # init target network
            brainstate.nn.init_all_states(self.target)
            # init etrace algorithm
            inp = jax.ShapeDtypeStruct(inputs.shape[2:], inputs.dtype)
            model.compile_graph(inp)
            model.show_graph()

        init()
        model = brainstate.nn.Vmap(model, vmap_states='new')

        warmup = self.args.warmup + inputs.shape[0] if self.args.warmup < 0 else self.args.warmup
        n_sim = int(warmup) if warmup > 0 else 0

        def _etrace_grad(inp):
            # call the model
            out = model(inp)
            # calculate the loss
            me, l1 = self._loss(out, targets)
            return me + l1, (out, me, l1)

        def _etrace_step(prev_grads, inp):
            # no need to return weights and states, since they are generated then no longer needed
            f_grad = brainstate.augment.grad(_etrace_grad,
                                             grad_states=self.trainable_weights,
                                             has_aux=True,
                                             return_value=True)
            cur_grads, local_loss, (out, mse_l, l1) = f_grad(inp)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, (out, mse_l, l1)

        def _etrace_train(inputs_):
            # forward propagation
            grads = jax.tree.map(jnp.zeros_like, self.trainable_weights.to_dict_values())
            grads, (outs, mse_ls, l1_ls) = brainstate.compile.scan(_etrace_step, grads, inputs_)
            acc = self._acc(outs, targets)

            grads = brainstate.functional.clip_grad_norm(grads, 1.)
            self.optimizer.update(grads)
            # accuracy
            return mse_ls.mean(), l1_ls.mean(), acc

        # running indices
        if n_sim > 0:
            brainstate.compile.for_loop(model, inputs[:n_sim])
            r = _etrace_train(inputs[n_sim:])
        else:
            r = _etrace_train(inputs)
        return r

    @brainstate.compile.jit(static_argnums=(0,))
    def bptt_train(self, inputs, targets) -> Tuple:
        inputs = jnp.asarray(inputs, dtype=brainstate.environ.dftype())
        # inputs: [n_seq, n_batch, n_feat]
        brainstate.nn.vmap_init_all_states(self.target, axis_size=inputs.shape[1], state_tag='new')
        model = brainstate.nn.Vmap(self.target, vmap_states='new')

        warmup = self.args.warmup + inputs.shape[0] if self.args.warmup < 0 else self.args.warmup
        n_sim = int(warmup) if warmup > 0 else 0

        def _step_run(inp):
            out = model(inp)
            return self._loss(out, targets), out

        def _bptt_grad():
            (mse_losses, l1_losses), outs = brainstate.compile.for_loop(_step_run, inputs)
            mse_losses = mse_losses[n_sim:].mean()
            l1_losses = l1_losses[n_sim:].mean()
            acc = self._acc(outs[n_sim:], targets)
            return mse_losses + l1_losses, (mse_losses, l1_losses, acc)

        f_grad = brainstate.augment.grad(_bptt_grad, grad_states=self.trainable_weights, has_aux=True, return_value=True)
        grads, loss, (mse_losses, l1_losses, acc) = f_grad()
        grads = brainstate.functional.clip_grad_norm(grads, 1.)
        self.optimizer.update(grads)
        return mse_losses, l1_losses, acc

    def f_sim(self):
        inputs, outputs = next(iter(self.loader))
        inputs = jnp.asarray(inputs, dtype=brainstate.environ.dftype()).transpose(1, 0, 2)
        self.target.visualize(inputs)

    def f_train(self):
        file = None
        if self.filepath is not None:
            os.makedirs(self.filepath, exist_ok=True)
            file = open(f'{self.filepath}/loss.txt', 'w')
        self.print(self.args, file=file)

        acc_max = 0.
        t0 = time.time()
        for bar_idx, (inputs, outputs) in enumerate(self.loader):
            if bar_idx > self.args.epochs:
                break

            inputs = jnp.asarray(inputs, dtype=brainstate.environ.dftype()).transpose(1, 0, 2)
            outputs = jnp.asarray(outputs, dtype=brainstate.environ.ditype())
            mse_ls, l1_ls, acc = (
                self.bptt_train(inputs, outputs)
                if self.args.method == 'bptt' else
                self.etrace_train(inputs, outputs)
            )
            if (bar_idx + 1) % 100 == 0:
                self.optimizer.lr.step_epoch()
            desc = (
                f'Batch {bar_idx:2d}, '
                f'CE={float(mse_ls):.8f}, '
                f'L1={float(l1_ls):.6f}, '
                f'acc={float(acc):.6f}, '
                f'time={time.time() - t0:.2f} s'
            )
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
    gargs = parse_args()
    brainstate.environ.set(dt=gargs.dt)

    # filepath
    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    exp_name = 'results/'
    if gargs.exp_name:
        exp_name += gargs.exp_name + '/'
    if gargs.method == 'bptt':
        filepath = f'{exp_name}/{gargs.method}/tau_a={gargs.tau_a}-tau_neu={gargs.tau_neu}-tau_syn={gargs.tau_syn}-{now}'
    else:
        filepath = f'{exp_name}/{gargs.method}/tau_a={gargs.tau_a}-tau_neu={gargs.tau_neu}-tau_syn={gargs.tau_syn}-{now}'

    # data
    with brainstate.environ.context(dt=brainstate.environ.get_dt() * u.ms):
        loader = EvidenceAccumulation(batch_size=gargs.batch_size)
    gargs.warmup = -(loader.t_recall / u.ms / brainstate.environ.get_dt())

    # network
    cls = SNNCobaNet if gargs.net == 'coba' else SNNCubaNet
    net = cls(
        loader.num_inputs,
        gargs.n_rec,
        loader.num_outputs,
        beta=gargs.beta,
        tau_a=gargs.tau_a,
        tau_neu=gargs.tau_neu,
        tau_syn=gargs.tau_syn,
        tau_out=gargs.tau_out,
        ff_scale=gargs.ff_scale,
        rec_scale=gargs.rec_scale,
        w_ei_ratio=gargs.w_ei_ratio,
        filepath=filepath,
        diff_spike=gargs.diff_spike,
        neuron_type=gargs.neuron_type,
    )

    # optimizer
    opt = brainstate.optim.Adam(lr=gargs.lr)

    # trainer
    trainer = Trainer(net, opt, loader, gargs, filepath=filepath)
    if gargs.mode == 'sim':
        trainer.f_sim()
    else:
        trainer.f_train()


if __name__ == '__main__':
    training()

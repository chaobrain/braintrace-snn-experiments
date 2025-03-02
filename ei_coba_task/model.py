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

import pickle
from typing import Callable

import brainstate
import braintools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import brainscale
from bst_utils import raster_plot


class GIF(brainstate.nn.Neuron):
    def __init__(
        self,
        size,
        V_rest: float = 0.,
        V_th_inf: float = 1.,
        tau: float = 20.,
        tau_a: float = 50.,
        beta: float = 1.,
        diff_spike: bool = True,
        V_initializer: Callable = brainstate.init.Uniform(0., 1.),
        I2_initializer: Callable = brainstate.init.ZeroInit(),
        spike_fun: Callable = brainstate.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(size, name=name, spk_fun=spike_fun, spk_reset=spk_reset)

        # params
        self.diff_spike = diff_spike
        self.V_rest = brainstate.init.param(V_rest, self.varshape, allow_none=False)
        self.V_th_inf = brainstate.init.param(V_th_inf, self.varshape, allow_none=False)
        self.tau = brainstate.init.param(tau, self.varshape, allow_none=False)
        self.tau_I2 = brainstate.init.param(tau_a, self.varshape, allow_none=False)
        self.beta = brainstate.init.param(beta, self.varshape, allow_none=False)

        # initializers
        self._V_initializer = V_initializer
        self._I_initializer = I2_initializer

    @property
    def num(self):
        return self.varshape[0]

    def init_state(self):
        self.V = brainscale.ETraceState(self._V_initializer(self.varshape))
        self.I_adp = brainscale.ETraceState(self._I_initializer(self.varshape))

    def dI2(self, I2, t):
        return - I2 / self.tau_I2

    def dV(self, V, t, I_ext):
        I_ext = self.sum_current_inputs(I_ext, V)
        return (- V + self.V_rest + I_ext) / self.tau

    def update(self, x=0.):
        last_spk = self.get_spike()
        if not self.diff_spike:
            last_spk = jax.lax.stop_gradient(last_spk)
        last_V = self.V.value - self.V_th_inf * last_spk
        last_I2 = self.I_adp.value - self.beta * last_spk

        I2 = brainstate.nn.exp_euler_step(self.dI2, last_I2, None)
        V = brainstate.nn.exp_euler_step(self.dV, last_V, None, I_ext=(x + I2))
        V = self.sum_delta_inputs(V)
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


class GIFv2(brainstate.nn.Neuron):
    def __init__(
        self,
        size,
        V_rest: float = 0.,
        V_th_inf: float = 1.,
        tau: float = 20.,
        tau_a: float = 50.,
        beta: float = 1.,
        diff_spike: bool = True,
        V_initializer: Callable = brainstate.init.Uniform(0., 1.),
        I2_initializer: Callable = brainstate.init.ZeroInit(),
        spike_fun: Callable = brainstate.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(size, name=name, spk_fun=spike_fun, spk_reset=spk_reset)

        # params
        self.diff_spike = diff_spike
        self.V_rest = brainstate.init.param(V_rest, self.varshape, allow_none=False)
        self.V_th_inf = brainstate.init.param(V_th_inf, self.varshape, allow_none=False)
        self.tau = brainstate.init.param(tau, self.varshape, allow_none=False)
        self.tau_I2 = brainstate.init.param(tau_a, self.varshape, allow_none=False)
        self.beta = brainstate.init.param(beta, self.varshape, allow_none=False)

        # initializers
        self._V_initializer = V_initializer
        self._I_initializer = I2_initializer

    @property
    def num(self):
        return self.varshape[0]

    def init_state(self):
        self.V = brainscale.ETraceState(self._V_initializer(self.varshape))
        self.I_adp = brainscale.ETraceState(self._I_initializer(self.varshape))
        self.spike = brainscale.ETraceState(jnp.zeros(self.varshape))

    def dI2(self, I2, t):
        return - I2 / self.tau_I2

    def dV(self, V, t, I_ext):
        I_ext = self.sum_current_inputs(I_ext, V)
        return (- V + self.V_rest + I_ext) / self.tau

    def update(self, x=0.):
        I2 = brainstate.nn.exp_euler_step(self.dI2, self.I_adp.value, None)
        V = brainstate.nn.exp_euler_step(self.dV, self.V.value, None, I_ext=(x + I2))
        V = self.sum_delta_inputs(V)

        spike = self.spk_fun((V - self.V_th_inf) / self.V_th_inf)
        if not self.diff_spike:
            spike = jax.lax.stop_gradient(spike)
        V = V - self.V_th_inf * spike
        I_adp = I2 - self.beta * spike

        self.I_adp.value = I_adp
        self.V.value = V
        self.spike.value = spike

        # outputs
        return jax.nn.standardize(V, axis=-1)

    def get_spike(self, V=None):
        if V is None:
            return self.spike.value
        else:
            return self.spk_fun((V - self.V_th_inf) / self.V_th_inf)


class _SNNEINet(brainstate.nn.Module):
    def __init__(
        self, n_in, n_rec, n_out, tau_neu=10., tau_a=100., beta=1., tau_syn=10., tau_out=10.,
        ff_scale=1., rec_scale=1., E_exc=None, E_inh=None, w_ei_ratio: float = 10., filepath=None,
        diff_spike=False
    ):
        super().__init__()

        self.filepath = filepath
        self.n_exc = int(n_rec * 0.8)
        self.n_inh = n_rec - self.n_exc

        # neurons
        tau_a = brainstate.random.uniform(100., tau_a * 2., n_rec)
        self.pop = GIF(n_rec, tau=tau_neu, tau_a=tau_a, beta=beta, diff_spike=diff_spike)
        ff_init = brainstate.init.KaimingNormal(scale=ff_scale)
        # feedforward
        self.ff2r = brainstate.nn.AlignPostProj(
            comm=brainscale.nn.SignedWLinear(n_in, n_rec, w_init=ff_init),
            syn=brainstate.nn.Expon.desc(in_size=n_rec, tau=tau_syn, g_initializer=brainstate.init.ZeroInit()),
            out=(brainstate.nn.CUBA.desc(scale=1.) if E_exc is None else brainstate.nn.COBA.desc(E=E_exc)),
            post=self.pop
        )
        # recurrent
        inh_init = brainstate.init.KaimingNormal(scale=rec_scale * w_ei_ratio)
        exc_init = brainstate.init.KaimingNormal(scale=rec_scale)
        inh2r_conn = brainscale.nn.SignedWLinear(self.n_inh, n_rec, w_init=inh_init,
                                                 w_sign=-1. if E_inh is None else None)
        exc2r_conn = brainscale.nn.SignedWLinear(self.n_exc, n_rec, w_init=exc_init)

        self.inh2r = brainstate.nn.AlignPostProj(
            comm=inh2r_conn,
            syn=brainscale.nn.Expon.desc(in_size=n_rec, tau=tau_syn, g_initializer=brainstate.init.ZeroInit()),
            out=(brainstate.nn.CUBA.desc(scale=1) if E_inh is None else brainstate.nn.COBA.desc(E=E_inh)),
            post=self.pop
        )
        self.exc2r = brainstate.nn.AlignPostProj(
            comm=exc2r_conn,
            syn=brainscale.nn.Expon.desc(in_size=n_rec, tau=tau_syn, g_initializer=brainstate.init.ZeroInit()),
            out=(brainstate.nn.CUBA.desc(scale=1) if E_exc is None else brainstate.nn.COBA.desc(E=E_exc)),
            post=self.pop
        )
        # output
        self.out = brainscale.nn.LeakyRateReadout(n_rec, n_out, tau=tau_out)

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

    @brainstate.compile.jit(static_argnums=0)
    def predict(self, batched_inputs):
        # batched_inputs: [n_seq, n_in]
        brainstate.nn.vmap_init_all_states(self, axis_size=batched_inputs.shape[1], state_tag='new')

        def step(inp):
            model = brainstate.nn.Vmap(self, vmap_states='new')
            out = model(inp)
            spk = self.pop.get_spike()
            rec_mem = self.pop.V.value
            return spk, rec_mem, out

        res = brainstate.compile.for_loop(step, batched_inputs, pbar=brainstate.compile.ProgressBar(10))
        return res

    def visualize(self, inputs, n2show: int = 5, filename: str = None):
        n_seq = inputs.shape[0]
        n_rec = self.pop.num
        indices = np.arange(0, n_rec, n_rec // 10)
        res = self.predict(inputs)
        res = {'rec_spk': res[0], 'rec_mem': res[1][..., indices], 'out': res[2]}

        indices = np.arange(n_seq)
        fig, gs = braintools.visualize.get_figure(4, n2show, 3., 4.5)
        for i in range(n2show):
            # input spikes
            raster_plot(indices, inputs[:, i], ax=fig.add_subplot(gs[0, i]), xlim=(0, n_seq))
            # recurrent spikes
            raster_plot(indices, res['rec_spk'][:, i], ax=fig.add_subplot(gs[1, i]), xlim=(0, n_seq))
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
        self,
        n_in,
        n_rec,
        n_out,
        tau_neu=10.,
        tau_a=100.,
        beta=1.,
        tau_syn=10.,
        tau_out=10.,
        ff_scale=1.,
        rec_scale=1.,
        w_ei_ratio=4.,
        filepath=None,
        diff_spike=False,
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
            diff_spike=diff_spike,
        )


class SNNCobaNet(_SNNEINet):
    def __init__(
        self,
        n_in,
        n_rec,
        n_out,
        tau_neu=10.,
        tau_a=100.,
        beta=1.,
        tau_syn=10.,
        tau_out=10.,
        ff_scale=1.,
        rec_scale=1.,
        w_ei_ratio=4.,
        filepath=None,
        diff_spike=False
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
            diff_spike=diff_spike,
        )

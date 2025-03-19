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

import brainscale
import brainstate
import braintools
import brainunit as u
import jax
import matplotlib.pyplot as plt
import numpy as np
import seaborn

from data import EvidenceAccumulation
from model import SNNCobaNet
from training import Trainer


def _plot_weight_image(weights, title=''):
    rec_weight = np.concatenate(
        [weights[('exc2r', 'comm', 'weight')].value['weight'],
         weights[('inh2r', 'comm', 'weight')].value['weight']]
    )
    ff_weight = np.asarray(weights[('ff2r', 'comm', 'weight')].value['weight'])
    fig, gs = braintools.visualize.get_figure(2, 1, 4.5, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    img = ax.imshow(rec_weight)
    plt.colorbar(img)
    ax.set_title('Recurrent weights')

    ax = fig.add_subplot(gs[1, 0])
    img = ax.imshow(ff_weight)
    plt.colorbar(img)
    ax.set_title('Feed-forward weights')
    plt.suptitle(title)
    plt.show()


def _plot_weight_hist(weights, title=''):
    fig, gs = braintools.visualize.get_figure(1, 3, 4.5, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    img = ax.hist(
        np.abs(np.asarray(weights[('exc2r', 'comm', 'weight')].value['weight']).flatten()),
        bins=50
    )
    ax.set_title('Recurrent weights [exc2r]')

    ax = fig.add_subplot(gs[0, 1])
    img = ax.hist(
        np.abs(np.asarray(weights[('inh2r', 'comm', 'weight')].value['weight']).flatten()),
        bins=50
    )
    ax.set_title('Recurrent weights [inh2r]')

    ax = fig.add_subplot(gs[0, 2])
    img = ax.hist(
        np.asarray(weights[('ff2r', 'comm', 'weight')].value['weight']).flatten(),
        bins=50
    )
    ax.set_title('Feed-forward weights')
    plt.suptitle(title)
    plt.show()


def _plot_sorted_spikes(spikes, title='', savepath=None, figsize=(4.5, 6.0)):
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

    fig, gs = braintools.visualize.get_figure(1, 1, *figsize)
    ax = fig.add_subplot(gs[0, 0])
    # Plot each spike, adjusting the neuron index based on the sorted order
    final_exc_times, final_exc_indices = [], []
    final_inh_times, final_inh_indices = [], []
    for time_, neuron in zip(spike_times, neuron_indices):
        sorted_index = np.where(sorted_neurons == neuron)[0][0]
        if neuron > 300:
            final_exc_times.append(time_)
            final_exc_indices.append(sorted_index)
            # ax.plot(time_, sorted_index, 'm.', markersize=2)
            ax.scatter(time_, sorted_index, color='m', s=3, marker='.')
        else:
            final_inh_times.append(time_)
            final_inh_indices.append(sorted_index)
            # ax.plot(time_, sorted_index, 'y.', markersize=2)
            ax.scatter(time_, sorted_index, color='y', s=3, marker='+')

    # ax.plot(final_exc_times, final_exc_indices, 'm.', markersize=2, label='Excitatory')
    # ax.plot(final_inh_times, final_inh_indices, 'y.', markersize=2, label='Inhibitory')
    # ax.legend(fontsize=12, fancybox=True, framealpha=0.5,  bbox_to_anchor=(1.0, 1.0))

    # ax.scatter(final_exc_times, final_exc_indices, color='m', s=3, marker='.', label='Excitatory')
    # ax.scatter(final_inh_times, final_inh_indices, color='y', s=3, marker='+', label='Inhibitory')
    # ax.legend(fontsize=12, fancybox=True, framealpha=0.5,  loc='best')

    # Adjusting the plot
    plt.ylabel('Neuron Index (sorted)')
    plt.yticks([])
    plt.xlim(-5, spikes.shape[0])
    # plt.ylim(-5, 130)
    plt.xticks(np.linspace(0, spikes.shape[0], 5))
    seaborn.despine()

    if title:
        plt.title(title)
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()


def _ploy_recurrent_membrane(mem, spk, savepath=None, figsize=(4.5, 6.0)):
    mem = np.where(spk, mem + 4, mem)

    fig, gs = braintools.visualize.get_figure(1, 1, *figsize)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(mem)
    plt.ylabel('Membrane (mV)')
    # plt.xlabel('Time (ms)')
    plt.xlim(-5, mem.shape[0])
    plt.xticks(np.linspace(0, mem.shape[0], 5))
    seaborn.despine()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()


def _plot_readout(out, savepath=None, figsize=(4.5, 6.0)):
    fig, gs = braintools.visualize.get_figure(1, 1, *figsize)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(out[:, 0], label='Right Decision')
    ax.plot(out[:, 1], label='Left Decision')
    plt.legend(loc='best')
    plt.xlim(-5, out.shape[0])
    plt.xticks(np.linspace(0, out.shape[0], 5))
    seaborn.despine()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()


def _plot_input_spikes(spikes, savepath=None, figsize=(4.5, 6.0)):
    def raster_plot(point, offset: int = 0):
        times, indices = np.where(point)
        plt.scatter(times, indices + offset, color='k', s=2., marker='o')

    xs = np.asarray(spikes)
    left_point = xs[:, :25]
    right_point = xs[:, 25:50]
    cue_point = xs[:, 50:75]
    noise_point = xs[:, 75:]

    fig, gs = braintools.visualize.get_figure(1, 1, *figsize)
    fig.add_subplot(gs[0, 0])
    raster_plot(left_point, offset=0)
    raster_plot(right_point, offset=27)  # 25 + 2
    raster_plot(cue_point, offset=54)  # 50 + 4
    raster_plot(noise_point, offset=81)  # 75 + 6
    plt.xlim(-5, cue_point.shape[0])
    plt.ylim(-5, 105)
    plt.xticks(np.linspace(0, cue_point.shape[0], 5))
    seaborn.despine()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()


def show_sorted_spikes_and_membrane():
    filepath = 'results-mem=200/esd-rtrl-multi-step-etrace=0.99/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-09 16-43-55'
    filepath = 'results-mem=200/esd-rtrl-multi-step-etrace=0.99/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-09 16-38-52'
    filepath = 'results-mem=200/d-rtrl-multi-step/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-09 16-43-16'
    filepath = 'results-mem=200/bptt/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-09 16-21-24'

    filepath = 'results-with-diff-spk-for-acc/esd-rtrl-multi-step-etrace=0.99/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-09 22-55-28'
    filepath = 'results-with-diff-spk-for-acc/d-rtrl-multi-step/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-10 07-27-35'
    filepath = 'results-with-diff-spk-for-acc/bptt/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-10 00-05-45'

    filepath = 'results-without-diff-spk-for-acc/d-rtrl-multi-step/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-11 02-48-16'
    filepath = 'results-without-diff-spk-for-acc/esd-rtrl-multi-step-etrace=0.99/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-10 13-25-55'

    with open(f'{filepath}/loss.txt', 'r') as f:
        print(f'Loading {filepath} ...')
        args = f.readline().strip().replace('Namespace', 'dict')
        args = brainstate.util.DotDict(eval(args))

    n2show = 5
    with brainstate.environ.context(dt=args.dt * u.ms):
        task = EvidenceAccumulation()
        keys = brainstate.random.RandomState(123).split_key(n2show)
        # keys = brainstate.random.split_key(n2show)
        xs = np.asarray(task.sampling(keys)[0])
        xs = np.transpose(xs, (1, 0, 2))

    with brainstate.environ.context(dt=args.dt):
        net = SNNCobaNet(task.num_inputs, args.n_rec, task.num_outputs, args)
        param_states = net.states(brainstate.ParamState)
        braintools.file.msgpack_load(f'{filepath}/best_model.msgpack', param_states)

        spk, rec_mem, out = net.predict(xs)

        _plot_input_spikes(
            xs[:, 0],
            savepath=f'{filepath}/input_spikes.svg',
            figsize=(3.5, 8)
        )
        _plot_sorted_spikes(
            spk[:, 0],
            savepath=f'{filepath}/sorted_spikes.svg',
            figsize=(3.5, 8)
        )
        indices = np.where(spk[:, 0].sum(0) == 2)[0]
        print(indices)
        # indices = np.asarray([9,  27,  47, 125, 128, 164,])
        # indices = np.asarray([32, 163, 250, 312, 405, 517, 779])
        _ploy_recurrent_membrane(
            rec_mem[:, 0, indices[:5]],
            spk[:, 0, indices[:5]],
            savepath=f'{filepath}/recurrent_membrane.svg',
            figsize=(3.5, 8)
        )
        _plot_readout(
            brainstate.functional.softmax(out[:, 0], axis=1),
            savepath=f'{filepath}/readout.svg',
            figsize=(1.5, 8)
        )


def show_esd_rtrl_eligibility_trace():
    filepath = 'results-mem=200/esd-rtrl-multi-step-etrace=0.99/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-09 16-43-55'
    filepath = 'results-mem=200/esd-rtrl-multi-step-etrace=0.99/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-09 16-38-52'
    filepath = 'results-mem=200/d-rtrl-multi-step/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-09 16-43-16'
    filepath = 'results-mem=200/bptt/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-09 16-21-24'
    filepath = 'results/esd-rtrl-multi-step-etrace=0.99/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-09 20-16-56'

    filepath = 'results/esd-rtrl-multi-step-etrace=0.99/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-09 22-55-28'
    with open(f'{filepath}/loss.txt', 'r') as f:
        print(f'Loading {filepath} ...')
        args = f.readline().strip().replace('Namespace', 'dict')
        args = brainstate.util.DotDict(eval(args))

    n2show = 5
    with brainstate.environ.context(dt=args.dt * u.ms):
        task = EvidenceAccumulation()
        keys = brainstate.random.RandomState(123).split_key(n2show)
        xs = np.asarray(task.sampling(keys)[0])
        xs = np.transpose(xs, (1, 0, 2))
        xs = np.asarray(xs, dtype=np.float32)

    with brainstate.environ.context(dt=args.dt):
        net = SNNCobaNet(task.num_inputs, args.n_rec, task.num_outputs, args)
        param_states = net.states(brainstate.ParamState)
        braintools.file.msgpack_load(f'{filepath}/best_model.msgpack', param_states)

        brainstate.nn.init_all_states(net)
        net = brainscale.IODimVjpAlgorithm(net, decay_or_rank=args.etrace_decay, vjp_method=args.vjp_method)
        net.compile_graph(jax.ShapeDtypeStruct(xs[0, 0].shape, dtype=xs.dtype))
        net.show_graph()

        def predict(x):
            net(x)
            return {k: v.value[0] for k, v in net.etrace_dfs.items()}

        etraces = brainstate.compile.for_loop(predict, xs[:, 0])
        for i, (k, v) in enumerate(etraces.items()):
            if i > 0:
                break
            for i in range(v.shape[1]):
                plt.plot(v[:, i])
        plt.legend()
        plt.show()

        # _plot_sorted_spikes(spk[:, 0], savepath=f'{filepath}/sorted_spikes.svg', figsize=(2, 12))
        # indices = np.asarray([32, 163, 250, 312, 405, 517, 779])
        # _ploy_recurrent_membrane(rec_mem[:, 0, indices],
        #                          spk[:, 0, indices],
        #                          savepath=f'{filepath}/recurrent_membrane.svg',
        #                          figsize=(2, 12))


def load_gif_esd_rtrl_model():
    filepath = 'results-mem=200/esd-rtrl-multi-step-etrace=0.99/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-09 16-43-55'
    filepath = 'results-mem=200/esd-rtrl-multi-step-etrace=0.99/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-09 16-38-52'
    # filepath = 'results-mem=200/d-rtrl-multi-step/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-09 16-43-16'
    # filepath = 'results-mem=200/bptt/tau_I1=50.0-A1=0.01-tau_I2=2000.0-A2=1.0-tau_neu=200.0-tau_syn=10.0-2025-03-09 16-21-24'
    with open(f'{filepath}/loss.txt', 'r') as f:
        print(f'Loading {filepath} ...')
        args = f.readline().strip().replace('Namespace', 'dict')
        args = brainstate.util.DotDict(eval(args))

    n2show = 5
    with brainstate.environ.context(dt=args.dt * u.ms):
        task = EvidenceAccumulation()
        keys = brainstate.random.RandomState(123).split_key(n2show)
        xs = np.asarray(task.sampling(keys)[0])
        xs = np.transpose(xs, (1, 0, 2))

    with brainstate.environ.context(dt=args.dt):
        net = SNNCobaNet(task.num_inputs, args.n_rec, task.num_outputs, args)
        param_states = net.states(brainstate.ParamState)
        braintools.file.msgpack_load(f'{filepath}/best_model.msgpack', param_states)

        spk, rec_mem, out = net.predict(xs)
        _plot_sorted_spikes(spk[:, 0])


if __name__ == '__main__':
    pass
    print(Trainer)

    # load_model()
    show_sorted_spikes_and_membrane()
    # show_esd_rtrl_eligibility_trace()
    # load_gif_esd_rtrl_model()

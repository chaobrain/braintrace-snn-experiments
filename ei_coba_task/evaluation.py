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

import brainpy_datasets as bd
import brainstate
import braintools
import matplotlib.pyplot as plt
import numpy as np

from model import SNNCobaNet


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
        fig, gs = braintools.visualize.get_figure(1, 2, 4.5, 6.0)
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

        fig, gs = braintools.visualize.get_figure(1, 1, 4.5, 6.0)
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

        fig, gs = braintools.visualize.get_figure(2, 1, 4.5 // 2, 6.0)
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

        fig, gs = braintools.visualize.get_figure(1, 1, 4.5, 6.0)
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

        fig, gs = braintools.visualize.get_figure(1, 1, 4.5, 6.0)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(times, index, 'k.', markersize=2)
        if savepath is not None:
            plt.savefig(savepath)
        else:
            plt.show()

    task = bd.cognitive.EvidenceAccumulation(dt=brainstate.environ.get_dt(), mode='spiking', )

    global gargs
    for key, filepath in different_path.items():
        with open(f'{filepath}/loss.txt', 'r') as f:
            print(f'Loading {filepath} ...')

            # parameters
            args = f.readline().strip().replace('Namespace', 'dict')
            gargs = brainstate.util.DotDict(eval(args))

            # environment
            brainstate.environ.set(
                mode=brainstate.mixin.JointMode(brainstate.mixin.Batching(), brainstate.mixin.Training()),
                dt=gargs.dt
            )
            brainstate.util.clear_name_cache()

            # task
            gargs.warmup = -(task.t_recall/u.ms / brainstate.environ.get_dt())
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
            gargs = brainstate.util.DotDict(eval(args))
            brainstate.environ.set(
                mode=brainstate.mixin.JointMode(brainstate.mixin.Batching(), brainstate.mixin.Training()), dt=gargs.dt)
            brainstate.util.clear_name_cache()
            gargs.warmup = -(task.t_recall / brainstate.environ.get_dt())
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

    task = bd.cognitive.EvidenceAccumulation(dt=brainstate.environ.get_dt(), mode='spiking', )
    ei_ratio1, firing_rates1 = get_ei_spiking_ratio(different_path['BPTT'])
    print('BPTT:', ei_ratio1)
    ei_ratio2, firing_rates2 = get_ei_spiking_ratio(different_path['D-RTRL'])
    print('D-RTRL:', ei_ratio2)
    ei_ratio3, firing_rates3 = get_ei_spiking_ratio(different_path['ES-D-RTRL'])
    print('ES-D-RTRL:', ei_ratio3)

    fig, gs = braintools.visualize.get_figure(1, 1, 3.0, 4.5)
    ax = fig.add_subplot(gs[0, 0])
    ax.violinplot([ei_ratio1, ei_ratio2, ei_ratio3], showmeans=False, showmedians=True)
    # ax.boxplot([ei_ratio1, ei_ratio2, ei_ratio3])
    ax.set_ylabel('E/I Spiking Ratio')
    plt.grid(True)
    ax.set_xticks([y + 1 for y in range(3)], labels=['BPTT', 'D-RTRL', 'ES-D-RTRL'])
    # plt.show()

    fig, gs = braintools.visualize.get_figure(1, 1, 3.0, 4.5)
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

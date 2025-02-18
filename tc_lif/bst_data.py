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

import os
import platform

import h5py
import jax.numpy as jnp
import numpy as np

__all__ = [
    'load_shd_dataset',
]


class SpikeIterator:
    def __init__(self, X, y, batch_size, nb_steps, nb_units, max_time, shuffle=True):
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        # self.max_time = max_time
        self.shuffle = shuffle
        self.labels_ = np.array(y, dtype=np.int_)
        self.num_samples = len(self.labels_)
        self.number_of_batches = np.ceil(self.num_samples / self.batch_size)
        self.sample_index = np.arange(len(self.labels_))
        # compute discrete firing times
        self.firing_times = X['times']
        self.units_fired = X['units']
        self.time_bins = np.linspace(0, max_time, num=nb_steps)
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.sample_index)
        self.counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_samples

    def __next__(self):
        if self.counter < self.number_of_batches:
            i_start = self.batch_size * self.counter
            i_end = min(self.batch_size * (self.counter + 1), self.num_samples)
            batch_index = self.sample_index[i_start:i_end]
            coo = [[] for i in range(3)]
            for bc, idx in enumerate(batch_index):
                times = np.digitize(self.firing_times[idx], self.time_bins)
                units = self.units_fired[idx]
                batch = [bc for _ in range(len(times))]

                coo[0].extend(times)
                coo[1].extend(batch)
                coo[2].extend(units)

            coo = list(map(np.asarray, coo))

            # [n_time, n_batch, n_input]
            X_batch = np.zeros([self.nb_steps, len(batch_index), self.nb_units])
            X_batch[*coo] = 1.0
            # [n_batch]
            y_batch = np.asarray(self.labels_[batch_index], dtype=jnp.int_)
            self.counter += 1
            return jnp.asarray(X_batch), jnp.asarray(y_batch)

        else:
            raise StopIteration


def _get_shd_data(dataset):
    # 雷神windows
    if platform.platform() == 'Windows-10-10.0.26100-SP0':
        root_path = 'D:/data/' + dataset + '/'

    # 雷神WSL
    if platform.platform() == 'Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.35':
        root_path = '/mnt/d/data/' + dataset + '/'

    # 吴思Lab A100
    if platform.platform() == 'Linux-6.8.0-52-generic-x86_64-with-glibc2.35':
        root_path = '/home/chaomingwang/data/' + dataset + '/'

    train_file = h5py.File(os.path.join(root_path, dataset.lower() + '_train.h5'), 'r')
    test_file = h5py.File(os.path.join(root_path, dataset.lower() + '_test.h5'), 'r')

    x_train = train_file['spikes']
    y_train = train_file['labels']
    x_test = test_file['spikes']
    y_test = test_file['labels']
    return (x_train, y_train), (x_test, y_test)


def load_shd_dataset(args):
    T = 250
    max_time = 1.4
    in_dim = 700
    (x_train, y_train), (x_test, y_test) = _get_shd_data(args.task)
    train_loader = SpikeIterator(x_train, y_train, args.batch_size, T, in_dim, max_time, shuffle=True)
    test_loader = SpikeIterator(x_test, y_test, args.batch_size, T, in_dim, max_time, shuffle=False)
    return train_loader, test_loader, in_dim

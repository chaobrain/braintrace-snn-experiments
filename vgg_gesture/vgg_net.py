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


from typing import Callable

import jax
import brainscale
import brainstate
import brainunit as u
import numpy as np


class LIFLayer(brainstate.nn.Neuron):
    def __init__(
        self,
        in_size,
        threshold: float = 0.6,
        spk_fun: Callable = brainstate.surrogate.ReluGrad(),
    ):
        super().__init__(in_size=in_size, spk_fun=spk_fun)

        # Fixed parameters
        self.threshold = threshold
        self.alpha_lim = [np.exp(-1 / 2), np.exp(-1 / 25)]

        # Trainable parameters
        self.alpha = brainscale.ElemWiseParam(
            brainstate.random.uniform(self.alpha_lim[0], self.alpha_lim[1], size=self.in_size),
        )

    def init_state(self, *args, **kwargs):
        self.ut = brainscale.ETraceState(brainstate.random.random(self.in_size))
        self.st = brainscale.ETraceState(brainstate.random.random(self.in_size))

    def update(self, x):
        # Compute spikes via neuron dynamics
        alpha = self.alpha.execute()
        alpha = u.math.clip(alpha, self.alpha_lim[0], self.alpha_lim[1])

        # Compute membrane potential (LIF)
        ut = alpha * self.ut.value - alpha * self.st.value + (1 - alpha) * x

        # Compute spikes with surrogate gradient
        st = self.spk_fun(ut - self.threshold)
        self.ut.value = ut
        self.st.value = st
        return st


class ConvLayer(brainstate.nn.Module):
    def __init__(self, in_size, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()

        self.conv2d = brainscale.nn.Conv2d(
            in_size, out_channels, kernel_size,
            stride=stride,
            padding=padding,
            w_init=brainstate.init.KaimingNormal(),
            b_init=brainstate.init.ZeroInit()
        )
        # self.lif = brainscale.nn.IF(self.conv2d.out_size, spk_fun=...)
        self.norm = brainstate.nn.LayerNorm(self.conv2d.out_size, )
        self.lif = LIFLayer(self.conv2d.out_size)
        self.in_size = in_size
        self.out_size = self.lif.out_size

    def update(self, x):
        x = self.conv2d(x)
        x = self.norm(x)
        x = self.lif(x)
        # jax.debug.print('spike count = {spk}', spk=x.sum())
        return x


class VGG(brainstate.nn.Module):
    def __init__(
        self,
        in_size: brainstate.typing.Size,
        n_label: int = 10,
        pool: str = 'AVG',
        dropout: float = 0.0,
        global_pool_size: int = 1,
    ):
        super().__init__()

        dropout = 1. - dropout
        pool_layer = brainstate.nn.AvgPool2d if pool == 'AVG' else brainstate.nn.MaxPool2d

        self.conv1 = ConvLayer(in_size, 64, kernel_size=3, padding=1, stride=1)
        self.dropout1 = brainstate.nn.DropoutFixed(self.conv1.out_size, dropout)

        self.conv2 = ConvLayer(self.dropout1.out_size, 128, kernel_size=3, padding=1, stride=1)
        self.dropout2 = brainstate.nn.DropoutFixed(self.conv2.out_size, dropout)
        self.pool1 = pool_layer(2, 2, in_size=self.dropout2.out_size)

        self.conv3 = ConvLayer(self.pool1.out_size, 256, kernel_size=3, padding=1, stride=1)
        self.dropout3 = brainstate.nn.DropoutFixed(self.conv3.out_size, dropout)

        self.conv4 = ConvLayer(self.dropout3.out_size, 256, kernel_size=3, padding=1, stride=1)
        self.dropout4 = brainstate.nn.DropoutFixed(self.conv4.out_size, dropout)
        self.pool2 = pool_layer(2, 2, in_size=self.dropout4.out_size)

        self.conv5 = ConvLayer(self.pool2.out_size, 512, kernel_size=3, padding=1, stride=1)
        self.dropout5 = brainstate.nn.DropoutFixed(self.conv5.out_size, dropout)

        self.conv6 = ConvLayer(self.dropout5.out_size, 512, kernel_size=3, padding=1, stride=1)
        self.dropout6 = brainstate.nn.DropoutFixed(self.conv6.out_size, dropout)
        self.pool3 = pool_layer(2, 2, in_size=self.dropout6.out_size)

        self.conv7 = ConvLayer(self.pool3.out_size, 512, kernel_size=3, padding=1, stride=1)
        self.dropout7 = brainstate.nn.DropoutFixed(self.conv7.out_size, dropout)

        self.conv8 = ConvLayer(self.dropout7.out_size, 512, kernel_size=3, padding=1, stride=1)
        self.dropout8 = brainstate.nn.DropoutFixed(self.conv8.out_size, dropout)

        self.global_pool = brainstate.nn.AdaptiveAvgPool2d((global_pool_size, global_pool_size),
                                                           in_size=self.dropout8.out_size)

        # self.readout = brainstate.nn.Linear(512 * global_pool_size * global_pool_size, n_label)
        self.readout = brainstate.nn.Linear([np.prod(self.global_pool.out_size)], n_label)

    def update(self, x):
        x = self.conv1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.dropout4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.dropout5(x)

        x = self.conv6(x)
        x = self.dropout6(x)
        x = self.pool3(x)

        x = self.conv7(x)
        x = self.dropout7(x)

        x = self.conv8(x)
        x = self.dropout8(x)

        x = self.global_pool(x)
        x = u.math.flatten(x)
        x = self.readout(x)

        return x


def dvs_vgg_stllr(args, in_size: brainstate.typing.Size):
    return VGG(
        in_size=in_size,  # Assuming input size is (height, width, channels)
        n_label=11,
        pool=args.pooling,
        dropout=args.dropout,
        global_pool_size=args.global_pool_size
    )

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

import brainstate
import brainunit as u


class GradSigmoid(brainstate.surrogate.Surrogate):
    gamma = 0.3

    # def surrogate_fun(self, x):
    #     pass

    def surrogate_grad(self, x):
        sgax = brainstate.functional.sigmoid(x * 4)
        surrogate = (1. - sgax) * sgax * 4
        return surrogate


class Surrogate(brainstate.surrogate.Surrogate):
    """
    Surrogate function for the gradient of the sigmoid activation function.
    """
    gamma = 0.3

    def surrogate_grad(self, vmem):
        thr = 0.6
        grad_x = self.gamma * u.math.max(
            u.math.zeros_like(vmem),
            1 - u.math.abs((vmem - thr) / thr)
        )
        return grad_x


class SurrogateAudio(brainstate.surrogate.Surrogate):
    gamma = 0.3

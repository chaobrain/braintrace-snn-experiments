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
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from bst_utils import MyArgumentParser


def parse_args():
    parser = MyArgumentParser()

    # Learning parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs.")

    # Dataset
    parser.add_argument("--task", type=str, default='ea', choices=['ea', 'dms'], help="")
    parser.add_argument("--batch_size", type=int, default=128, help="")
    parser.add_argument("--warmup", type=float, default=0., help="The ratio for network simulation.")

    # Model
    parser.add_argument("--neuron", type=str, default='gifv2', help="str: neuron type")
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
    return gargs

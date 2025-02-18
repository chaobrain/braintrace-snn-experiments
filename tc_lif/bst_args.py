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


import argparse
import os


def _set_gpu_preallocation(mode: float):
    """GPU memory allocation.

    If preallocation is enabled, this makes JAX preallocate ``percent`` of the total GPU memory,
    instead of the default 75%. Lowering the amount preallocated can fix OOMs that occur when the JAX program starts.
    """
    assert isinstance(mode, float) and 0. <= mode < 1., f'GPU memory preallocation must be in [0., 1.]. But got {mode}.'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(mode)


def _set_gpu_device(device_ids):
    if isinstance(device_ids, int):
        device_ids = str(device_ids)
    elif isinstance(device_ids, (tuple, list)):
        device_ids = ','.join([str(d) for d in device_ids])
    elif isinstance(device_ids, str):
        device_ids = device_ids
    else:
        raise ValueError
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99'


def parse_args(gpu_pre_allocate=0.99):
    parser = argparse.ArgumentParser(description='Sequential SHD/SSC')

    parser.add_argument('--devices', type=str, default='0', help='The GPU device ids.')
    parser.add_argument("--method", type=str, default='bptt', help="Training method.",
                        choices=['bptt', 'd-rtrl', 'esd-rtrl'])
    args, _ = parser.parse_known_args()

    # device management
    _set_gpu_device(args.devices)
    _set_gpu_preallocation(gpu_pre_allocate)

    # training method
    if args.method != 'bptt':
        parser.add_argument("--vjp_method", type=str, default='multi-step',
                            choices=['multi-step', 'single-step'])
        if args.method != 'd-rtrl':
            parser.add_argument(
                "--etrace_decay", type=float, default=0.99,
                help="The time constant of eligibility trace "
            )

    parser.add_argument('--task', default='shd', type=str, help='SHD, SSC', choices=['shd', 'ssc'])
    args, _ = parser.parse_known_args()
    if args.task == 'shd':
        parser.add_argument("--num-input", type=int, default=700)
        parser.add_argument('--num-hidden', default=128, type=int, help='hidden size')
        parser.add_argument("--num-output", type=int, default=20)
    elif args.task == 'ssc':
        parser.add_argument("--num-input", type=int, default=700)
        parser.add_argument('--num-hidden', default=400, type=int, help='hidden size')
        parser.add_argument("--num-output", type=int, default=35)
    else:
        raise ValueError('Invalid task choice. Please choose between SHD and SSC.')

    parser.add_argument('--optim', default='adam', type=str, help='optimizer (default: adam)')
    parser.add_argument('--results-dir', default='', type=str, metavar='PATH',
                        help='path to cache (default: none)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=0, type=int, metavar='N', help='seed')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate',
                        default=0.0005, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--schedule', default=[40, 80], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
    parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--wd', default=0, type=float, metavar='W', help='weight decay')
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument('--cos', action='store_true', default=False, help='use cosine lr schedule')
    parser.add_argument('--warmup-ratio', type=float, default=0.)

    # options for SNNs
    parser.add_argument('--time-window', default=250, type=int, help='')
    parser.add_argument('--threshold', default=1.5, type=float, help='')
    parser.add_argument('--detach-reset', action='store_true', default=False, help='')
    parser.add_argument('--hard-reset', action='store_true', default=False, help='')
    parser.add_argument('--decay-factor', default=1.0, type=float, help='')
    parser.add_argument('--beta1', default=0., type=float, help='')
    parser.add_argument('--beta2', default=0., type=float, help='')
    parser.add_argument('--gamma', default=0.5, type=float, help='dendritic reset scaling hyper-parameter')
    parser.add_argument('--sg', default='triangle', type=str, help='surrogate gradient: triangle and exp')
    parser.add_argument('--neuron', default='tclif', type=str, help='plif, lif, tclif')
    parser.add_argument('--network', default='ff', type=str, help='fb, ff')

    args = parser.parse_args()

    return args

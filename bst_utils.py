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

import glob
import logging
import os
import shutil
import sys

import braintools
import brainstate


def copy_files(tar_dir, dest_dir):
    for filename in glob.glob(os.path.join(tar_dir, '*.py')):
        print(filename)
        shutil.copy(filename, dest_dir, follow_symlinks=True)


def save_model_states(
    save_path: str,
    model: brainstate.nn.Module,
    optimizer: brainstate.optim.Optimizer,
    **kwargs
):
    """
    Save the current state of the model, optimizer, and training progress.

    This function creates a dictionary containing the current epoch, accuracy,
    model state, and optimizer state, then saves it to a file using MessagePack format.

    Parameters:
    -----------
    model : brainstate.nn.Module
        The neural network model whose state is to be saved.
    optimizer : brainstate.optim.Optimizer
        The optimizer used for training, whose state is to be saved.
    epoch : int
        The current epoch number.
    accuracy : float
        The current accuracy of the model.
    save_path : str
        The file path where the model state will be saved.

    Returns:
    --------
    None
        This function doesn't return any value, but it saves the state to a file
        and prints a confirmation message.
    """
    state = {
        'state_dict': model.states(brainstate.ParamState),
        'optimizer_state_dict': brainstate.graph.states(optimizer),
        **kwargs
    }
    braintools.file.msgpack_save(save_path, state)


def setup_logging(log_file: str) -> logging.Logger:
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)  # Set the minimum logging level

    # Create a formatter to customize the log message format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create a StreamHandler to output to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.WARNING)  # Set the logging level for stdout
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Create a FileHandler to output to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.WARNING)  # Set the logging level for the file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', unit=''):
        self.name = name
        self.fmt = fmt
        self.unit = unit
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}' + self.unit + ' ({avg' + self.fmt + '}' + self.unit + ')'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return ', '.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

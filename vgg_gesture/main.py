# -*- coding: utf-8 -*-

import argparse
import logging
import os

from train import Trainer
from dataloader import dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, default='./experiments/default', help='Directory to save the checkpoint and logs of the experiment')
    parser.add_argument('--data-path', type=str, default='../data', help='Path for the datasets folder. The datasets is going to be downloaded if it is not in the location.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'NAG', 'Adam', 'RMSProp', 'RProp'], default='Adam', help='Choice of the optimizer')
    parser.add_argument('--loss', type=str, choices=['MSE', 'CE'], default='CE', help='Choice of the loss function')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--etrace_decay', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=16, help='Batch size for testing')
    parser.add_argument('--label-encoding', type=str, default="class", choices=["class", "one-hot"],  help='Label encoding by default class. But, one-hot should be use for DFA.')
    parser.add_argument('--seed', type=int, default=None, help='Seed for reproducibility.')
    parser.add_argument('--method', type=str, default='bptt', choices=["bptt", "d-rtrl", "es-d-rtrl"], help='Training mode.')
    parser.add_argument('--vjp_method', type=str, default='multi-step', choices=["multi-step", "single-step"])
    parser.add_argument('--delay-ls', type=int, default=5, help='Number of time steps for which the learning signal is available (T - T_l).')
    parser.add_argument('--scheduler', type=int, default=0, help='Learning rate decay time.')
    parser.add_argument('--print-freq', type=int, default=200, help='Frequency of printing results.')
    parser.add_argument('--pooling', type=str, default='MAX', choices=["MAX", "AVG"], help='Pooling layer.')
    parser.add_argument('--weight-decay', type=float, default=0, help='Weight decay L2 normalization')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--global_pool_size', type=int, default=1, help='global pool size')
    args = parser.parse_args()

    # Create a new folder in 'args.save_path' to save the results of the experiment
    os.makedirs(args.save_path, exist_ok=True)

    # Log configuration
    log_path = args.save_path + "/log.log"
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filename=log_path)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(args)
    logging.info('=> Everything will be saved to {}'.format(args.save_path))

    # Initiate the training
    train_loader, test_loader = dataloader(args)
    Trainer(args=args).f_train(train_loader, test_loader)


if __name__ == '__main__':
    main()

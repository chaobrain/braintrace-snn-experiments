# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from general_utils import MyArgumentParser


def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"invalid truth value {val}")


def add_training_options(parser):
    parser.add_argument(
        "--load_exp_folder",
        type=str,
        default=None,
        help="Path to experiment folder with a pretrained model to load. Note "
             "that the same path will be used to store the current experiment."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='train',
    )
    parser.add_argument(
        "--new_exp_folder",
        type=str,
        default=None,
        help="Path to output folder to store experiment."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["shd", "ssc", "gesture", "gesturev2", "nmnist", "nmnistv2"],
        default="shd",
        help="Dataset name (shd, ssc, hd or sc)."
    )
    args, _ = parser.parse_known_args()

    parser.add_argument(
        '--data_length',
        type=int,
        default=100
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/SHD/')),
        help="Path to dataset folder.",
    )
    parser.add_argument(
        "--save_best",
        type=lambda x: bool(strtobool(str(x))), default=True,
        help="If True, the model from the epoch with the highest validation "
             "accuracy is saved, if False, no model is saved."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of input examples inside a single batch.",
    )
    parser.add_argument(
        "--nb_epochs",
        type=int,
        default=5,
        help="Number of training epochs (i.e. passes through the dataset)."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of training epochs (i.e. passes through the dataset)."
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="Epoch number to start training at. Will be 0 if no pretrained "
             "model is given. First epoch will be start_epoch+1."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Initial learning rate for training. The default value of 0.01 "
             "is good for SHD and SC, but 0.001 seemed to work better for HD and SC."
    )
    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=40,
        help="Number of epochs without progress before the learning rate gets decreased."
    )
    parser.add_argument(
        "--lr_step_gamma",
        type=float,
        default=0.9,
        help="Factor between 0 and 1 by which the learning rate gets "
             "decreased when the scheduler patience is reached."
    )
    parser.add_argument(
        "--use_augm",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="Whether to use data augmentation or not. Only implemented for "
             "non-spiking HD and SC datasets."
    )
    return parser


def add_model_options(parser):
    parser.add_argument(
        "--model_type",
        type=str,
        default="LIF",
        help="Type of ANN or SNN model.",
        choices=["LIF", "adLIF", "RLIF", "RadLIF", "MLP", "RNN", "LiGRU", "GRU"],
    )
    parser.add_argument(
        "--surrogate",
        type=str,
        default="boxcar",
    )
    parser.add_argument(
        "--nb_layers",
        type=int,
        default=3,
        help="Number of layers (including readout layer)."
    )
    parser.add_argument(
        "--nb_hiddens",
        type=int,
        default=128,
        help="Number of neurons in all hidden layers."
    )
    parser.add_argument(
        "--pdrop",
        type=float,
        default=0.1,
        help="Dropout rate, must be between 0 and 1."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--inp_scale",
        type=float,
        default=20.
    )
    parser.add_argument(
        "--rec_scale",
        type=float,
        default=5.0
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="none",
        choices=["none", "batchnorm", "layernorm", 'dyt', 'rmsnorm', 'weightnorm'],
        help="Type of normalization, every string different from batchnorm "
    )
    parser.add_argument(
        "--use_bias",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="Whether to include trainable bias with feedforward weights."
    )
    return parser


def parse_args():
    parser = MyArgumentParser(
        description="Model training on spiking speech commands datasets.",
        method='esd-rtrl'
    )
    parser = add_model_options(parser)
    parser = add_training_options(parser)
    args = parser.parse_args()
    return args


def main():
    """
    Runs model training/testing using the configuration specified
    by the parser arguments. Run `python main.py -h` for details.
    """

    # Get experiment configuration from parser
    args = parse_args()

    # Instantiate class for the desired experiment
    from models import Experiment
    experiment = Experiment(args)

    # Run experiment
    if args.mode == 'train':
        experiment.f_train()
    elif args.mode == 'test':
        experiment.f_test(8)
    else:
        raise ValueError("Mode must be either 'train' or 'test'")


if __name__ == "__main__":
    main()

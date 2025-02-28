import json
import logging
import math
import os
import sys
import time
from copy import deepcopy
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import jax
import jax.numpy as jnp
import optax

import brainunit as u
import brainscale
import brainstate
import braintools
from data import load_mnist_dataset
from models import get_neuron, FFNetMnist, RecNetMnist
from bst_utils import save_model_states, AverageMeter, ProgressMeter, setup_logging, MyArgumentParser


def parse_args():
    parser = MyArgumentParser(description='Sequential SHD/SSC')

    parser.add_argument('--task', default='SMNIST', type=str, choices=['SMNIST', 'PSMNIST'])
    args, _ = parser.parse_known_args()
    parser.add_argument("--num-input", type=int, default=1)
    parser.add_argument('--optim', default='adam', type=str, help='optimizer (default: adam)')
    parser.add_argument('--results-dir', default='', type=str, metavar='PATH')
    parser.add_argument('--data-dir', default='./data', type=str, metavar='PATH')
    parser.add_argument('--print-freq', default=10, type=int)
    parser.add_argument('--seed', default=0, type=int, metavar='N', help='seed')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, dest='lr')
    parser.add_argument('--schedule', default=[40, 80], nargs='*', type=int)
    parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--wd', default=0, type=float, metavar='W', help='weight decay')
    parser.add_argument('--warmup-ratio', type=float, default=0.)

    # options for SNNs
    if args.task == 'SMNIST':
        parser.add_argument('--threshold', default=1.0, type=float, help='')
    else:
        parser.add_argument('--threshold', default=1.8, type=float, help='')
    parser.add_argument('--detach-reset', action='store_true', default=False, help='')
    parser.add_argument('--hard-reset', action='store_true', default=False, help='')
    parser.add_argument('--decay-factor', default=1.0, type=float, help='')
    parser.add_argument('--beta1', default=0.0, type=float, help='')
    parser.add_argument('--beta2', default=0.0, type=float, help='')
    parser.add_argument('--gamma', default=0.5, type=float, help='dendritic reset scaling hyper-parameter')
    parser.add_argument('--sg', default='triangle', type=str, help='surrogate gradient: triangle and exp')
    parser.add_argument('--neuron', default='tclif', type=str, help='plif, lif, tclif')
    parser.add_argument('--network', default='ff', type=str, help='fb, ff')

    args = parser.parse_args()
    return args


class Trainer(brainstate.util.PrettyObject):
    def __init__(self, args):
        self.args = deepcopy(args)
        brainstate.random.seed(self.args.seed)

        self.perm = brainstate.random.permutation(jnp.arange(784))

        if self.args.results_dir == '':
            if self.args.method == 'esd-rtrl':
                self.args.results_dir = (
                    f'./exp/{self.args.task}-{self.args.network}'
                    f'-{self.args.method}-{self.args.etrace_decay}'
                    f'-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
                )
            else:
                self.args.results_dir = (
                    f'./exp/{self.args.task}-{self.args.network}-{self.args.method}'
                    f'-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
                )

        os.makedirs(self.args.results_dir, exist_ok=True)
        filename = "log-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".txt"
        self.logger = setup_logging(os.path.join(self.args.results_dir, filename))

        # model
        if self.args.network == 'ff':
            model_type = FFNetMnist
        elif self.args.network == 'fb':
            model_type = RecNetMnist
        else:
            raise NotImplementedError
        spiking_neuron = get_neuron(self.args)
        self.model = model_type(
            in_dim=self.args.num_input,
            spiking_neuron=spiking_neuron
        )
        table, _ = brainstate.nn.count_parameters(self.model, return_table=True)

        # logger
        self.logger.warning(str(self.model))
        self.logger.warning(str(table))

        # dump args
        with open(self.args.results_dir + '/args.json', 'w') as fid:
            json.dump(self.args.__dict__, fid, indent=2)
        logging.info(str(self.args))

        # optimizer
        # lr = optax.step_lr_schedule(
        #     init_value=self.args.lr,
        #     step_size=10,
        #     gamma=0.5
        # )
        lr = self.args.lr
        if self.args.optim == 'sgd':
            sgd_transform = optax.sgd(learning_rate=lr, momentum=0.9)
            weight_decay_transform = optax.add_decayed_weights(self.args.wd)
            chained_transform = optax.chain(weight_decay_transform, sgd_transform)
            optimizer = brainstate.optim.OptaxOptimizer(chained_transform)

        elif self.args.optim == 'adam':
            transform = optax.adamw(learning_rate=lr, weight_decay=self.args.wd)
            optimizer = brainstate.optim.OptaxOptimizer(transform)

        else:
            raise NotImplementedError
        self.trainable_weights = self.model.states(brainstate.ParamState)
        optimizer.register_trainable_weights(self.trainable_weights)
        self.optimizer = optimizer

        # best accuracy
        self.best_acc_top1 = 0.
        self.best_acc_top5 = 0.

    def _loss(self, predictions, targets):
        return braintools.metric.softmax_cross_entropy_with_integer_labels(predictions, targets).mean()

    def _accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions
        for the specified values of k"""
        maxk = max(topk)
        batch_size = target.shape[0]

        pred = jnp.argsort(output, axis=1)[:, -maxk:][:, ::-1]
        correct = pred == target[:, None]

        res = []
        for k in topk:
            correct_k = correct[:, :k].reshape(-1).sum(axis=0, keepdims=True)
            res.append(correct_k * 100.0 / batch_size)
        return res

    def _process_input(self, inputs):
        inputs = u.math.flatten(jnp.asarray(inputs), start_axis=1)
        inputs = inputs.transpose((1, 0))  # [n_time, n_batch, n_feature]
        if self.args.task == 'PSMNIST':
            inputs = inputs[self.perm]
        inputs = inputs.reshape(inputs.shape[0], -1, self.args.num_input)
        return inputs

    @brainstate.compile.jit(static_argnums=0)
    def predict(self, inputs: jax.Array, targets: jax.Array):
        inputs = self._process_input(inputs)

        # add environment context
        model = brainstate.nn.EnvironContext(self.model, fit=False)

        # assume the inputs have shape (time, batch, features, ...)
        n_time, n_batch = inputs.shape[:2]
        brainstate.nn.vmap_init_all_states(model, state_tag='hidden', axis_size=n_batch)

        def _step(inp):
            out = brainstate.augment.vmap(model, in_states=model.states('hidden'))(inp)
            return out

        # forward propagation
        outs = brainstate.compile.for_loop(_step, inputs)
        outs = outs.sum(axis=0)

        # loss
        loss = self._loss(outs, targets)

        # accuracy
        acc1, acc5 = self._accuracy(outs, targets, topk=(1, 5))
        return acc1, acc5, loss

    @brainstate.compile.jit(static_argnums=0)
    def bptt_train(self, inputs, targets):
        inputs = self._process_input(inputs)

        brainstate.nn.vmap_init_all_states(self.model, state_tag='hidden', axis_size=inputs.shape[1])
        model = brainstate.nn.EnvironContext(self.model, fit=True)
        model = brainstate.nn.Vmap(model, vmap_states='hidden')

        def _bptt_grad_step():
            outs = brainstate.compile.for_loop(model, inputs)
            outs = outs.sum(axis=0)
            loss = self._loss(outs, targets)
            return loss, outs

        # gradients
        grads, loss, out_sum = brainstate.augment.grad(
            _bptt_grad_step,
            self.trainable_weights,
            has_aux=True,
            return_value=True
        )()

        # optimization
        grads = brainstate.functional.clip_grad_norm(grads, 1.)
        self.optimizer.update(grads)

        # accuracy
        acc1, acc5 = self._accuracy(out_sum, targets, topk=(1, 5))

        return acc1, acc5, loss

    @brainstate.compile.jit(static_argnums=0)
    def online_train(self, inputs, targets):
        inputs = self._process_input(inputs)

        # assume the inputs have shape (time, batch, features, ...)
        n_time, n_batch = inputs.shape[:2]

        # initialize the online learning model
        model = brainstate.nn.EnvironContext(self.model, fit=True)
        if self.args.method == 'esd-rtrl':
            model = brainscale.IODimVjpAlgorithm(model, self.args.etrace_decay, vjp_method=self.args.vjp_method)
        elif self.args.method == 'd-rtrl':
            model = brainscale.ParamDimVjpAlgorithm(model, vjp_method=self.args.vjp_method)
        else:
            raise ValueError(f'Unknown online learning methods: {self.args.method}.')

        @brainstate.augment.vmap_new_states(state_tag='new', axis_size=n_batch)
        def init():
            """
            Initialize the model states and compile the computation graph.

            This function performs the following tasks:
            1. Creates a shape and dtype structure for the input.
            2. Initializes all states of the model.
            3. Compiles the computation graph.
            4. Displays the compiled graph.

            The function is decorated with `vmap_new_state`, which vectorizes the function
            across a new state axis with the tag 'new' and size `n_batch`.
            """
            inp = jax.ShapeDtypeStruct(inputs.shape[2:], inputs.dtype)
            brainstate.nn.init_all_states(self.model)
            model.compile_graph(inp)
            model.show_graph()

        init()
        model = brainstate.nn.Vmap(model, vmap_states='new')

        def _etrace_grad(inp):
            out = model(inp)
            loss = self._loss(out, targets)
            return loss, out

        def _etrace_step(prev_grads, x):
            # no need to return weights and states, since they are generated then no longer needed
            f_grad = brainstate.augment.grad(
                _etrace_grad,
                self.trainable_weights,
                has_aux=True,
                return_value=True
            )
            cur_grads, local_loss, out = f_grad(x)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, (out, local_loss)

        def _etrace_train(inputs_):
            # forward propagation
            grads = jax.tree.map(lambda a: jnp.zeros_like(a), self.trainable_weights.to_dict_values())
            grads, (outs, losses) = brainstate.compile.scan(_etrace_step, grads, inputs_)
            # gradient updates
            grads = brainstate.functional.clip_grad_norm(grads, 1.)
            self.optimizer.update(grads)
            # accuracy
            return losses.mean(), outs.sum(axis=0)

        loss, out_sum = _etrace_train(inputs)

        # accuracy
        acc1, acc5 = self._accuracy(out_sum, targets, topk=(1, 5))

        # returns
        return acc1, acc5, loss

    def train_epoch(self, train_loader, epoch):
        batch_time = AverageMeter('Time', ':6.3f', unit='s')
        data_time = AverageMeter('Data', ':6.3f', unit='s')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f', unit='%')
        top5 = AverageMeter('Acc@5', ':6.2f', unit='%')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch)
        )

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # training
            images = jnp.asarray(images)  # images:[bs, 1, 28, 28]
            target = jnp.asarray(target)
            n_batch = images.shape[0]
            if self.args.method == 'bptt':
                acc1, acc5, loss = self.bptt_train(images, target)
            else:
                acc1, acc5, loss = self.online_train(images, target)

            # measure accuracy and record loss
            losses.update(loss.item(), n_batch)
            top1.update(acc1[0], n_batch)
            top5.update(acc5[0], n_batch)
            batch_time.update(time.time() - end)

            # measure elapsed time
            end = time.time()
            if (i + 1) % self.args.print_freq == 0:
                self.logger.warning(progress.display(i))

        self.logger.warning(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        # self.logger.flush()
        return top1.avg, losses.avg

    def validate_epoch(self, val_loader):
        batch_time = AverageMeter('Time', ':6.3f', unit='s')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f', unit='%')
        top5 = AverageMeter('Acc@5', ':6.2f', unit='%')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: '
        )

        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # prediction
            images = jnp.asarray(images)  # images:[bs, 1, 28, 28]
            target = jnp.asarray(target)
            n_batch = images.shape[0]
            acc1, acc5, loss = self.predict(images, target)

            # measure accuracy and record loss
            losses.update(loss.item(), n_batch)
            top1.update(acc1[0], n_batch)
            top5.update(acc5[0], n_batch)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.args.print_freq == 0:
                self.logger.warning(progress.display(i))
        self.logger.warning(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        # self.logger.flush()
        return top1.avg, top5.avg, losses.avg

    def f_train(self, train_loader, test_loader):
        start_epoch = 0
        if self.args.print_freq > len(train_loader):
            self.args.print_freq = math.ceil(len(train_loader) // 2)

        for epoch in range(start_epoch, self.args.epochs):
            train_acc, train_loss = self.train_epoch(train_loader, epoch)
            acc1, acc5, test_loss = self.validate_epoch(test_loader)

            self.logger.warning(
                f'Test Epoch: [{epoch}/{self.args.epochs}], '
                f'train acc: {train_acc:.4f}, '
                f'train loss: {train_loss:.4f}, '
                f'test acc (top 1): {acc1:.4f}, '
                f'test acc (top 5): {acc5:.4f}, '
                f'test loss: {test_loss:.4f}'
            )
            self.logger.warning(
                f'Test Epoch: [{epoch}/{self.args.epochs}], '
                f'train top1 best acc: {self.best_acc_top1:.4f}, '
                f'train top5 best acc: {self.best_acc_top5:.4f}, '
            )
            # self.logger.flush()

            saved = False
            if self.best_acc_top1 < acc1:
                save_model_states(
                    os.path.join(self.args.results_dir, f'checkpoint-at-epoch-{epoch}-top1acc-{acc1:.5f}.msgpack'),
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    top1_accuracy=acc1,
                    top5_accuracy=acc5,
                )
                saved = True
                self.best_acc_top1 = acc1
            if self.best_acc_top5 < acc5:
                self.best_acc_top5 = acc5
                if not saved:
                    save_model_states(
                        os.path.join(self.args.results_dir, f'checkpoint-at-epoch-{epoch}-top5acc-{acc5:.5f}.msgpack'),
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch + 1,
                        top1_accuracy=acc1,
                        top5_accuracy=acc5,
                    )


def main():
    args = parse_args()

    # get datasets and build dataloaders
    train_loader, test_loader = load_mnist_dataset(args)

    # trainer and training
    trainer = Trainer(args)
    trainer.f_train(train_loader, test_loader)


if __name__ == "__main__":
    main()

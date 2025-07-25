# -*- coding: utf-8 -*-

import datetime
import logging
import os.path
import time

import brainscale
import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import optax

from metrics import AverageMeter, ProgressMeter, accuracy
from vgg_net import dvs_vgg_stllr


class Trainer:
    def __init__(self, args):
        self.args = args
        self.args.save_path = os.path.join(args.save_path, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

        # random seed for reproducibility
        if args.seed != 0:
            brainstate.random.seed(args.seed)

        # Network topology
        model = dvs_vgg_stllr(args, (32, 32, 2))
        brainstate.nn.count_parameters(model)
        self.model = model

        # Initial monitoring
        logging.info("=== Model ===")
        logging.info(model)

        # Optimizer
        self.weights = model.states(brainstate.ParamState)
        if args.optimizer == 'SGD':
            optimizer = optax.sgd(args.lr)
        elif args.optimizer == 'Adam':
            optimizer = optax.adam(args.lr)
        elif args.optimizer == 'NAG':
            optimizer = optax.sgd(args.lr, momentum=0.9, nesterov=True)
        elif args.optimizer == 'RMSprop':
            optimizer = optax.rmsprop(args.lr)
        else:
            raise NameError("=== ERROR: optimizer " + str(args.optimizer) + " not supported")
        self.optimizer = brainstate.optim.OptaxOptimizer(
            optax.chain(optax.add_decayed_weights(args.weight_decay), optimizer)
        )
        self.optimizer.register_trainable_weights(self.weights)

    def f_loss(self, predictions, targets):
        if self.args.loss == 'MSE':
            loss = braintools.metric.squared_error(predictions, targets)
        elif self.args.loss == 'CE':
            loss = braintools.metric.softmax_cross_entropy_with_integer_labels(predictions, targets)
        else:
            raise NameError("=== ERROR: loss " + str(self.args.loss) + " not supported")
        return loss.mean()

    @brainstate.transform.jit(static_argnums=0)
    def bptt_train(self, inputs, targets, labels):
        model = brainstate.nn.EnvironContext(self.model, fit=True)
        brainstate.nn.vmap_init_all_states(model, state_tag='hidden', axis_size=inputs.shape[1])
        model = brainstate.nn.Vmap(model, vmap_states='hidden')

        def _bptt_grad_step():
            outs = brainstate.transform.for_loop(model, inputs)
            outs = outs.sum(axis=0)
            loss = self.f_loss(outs, targets)
            return loss, outs

        # gradients
        grads, loss, outs = brainstate.transform.grad(_bptt_grad_step, self.weights, has_aux=True, return_value=True)()

        # optimization
        self.optimizer.update(grads)

        # accuracy
        acc1, acc5 = accuracy(outs, labels, topk=(1, 5))
        return acc1, acc5, loss

    @brainstate.compile.jit(static_argnums=0)
    def online_train(self, inputs, targets, labels):
        # assume the inputs have shape (time, batch, features, ...)
        n_time, n_batch = inputs.shape[:2]

        # initialize the online learning model
        model = brainstate.nn.EnvironContext(self.model, fit=True)
        if self.args.method == 'es-d-rtrl':
            model = brainscale.IODimVjpAlgorithm(model, self.args.etrace_decay, vjp_method=self.args.vjp_method)
        elif self.args.method == 'd-rtrl':
            model = brainscale.ParamDimVjpAlgorithm(model, vjp_method=self.args.vjp_method)
        else:
            raise ValueError(f'Unknown online learning methods: {self.args.method}.')

        @brainstate.augment.vmap_new_states(state_tag='new', axis_size=n_batch)
        def init():
            inp = jax.ShapeDtypeStruct(inputs.shape[2:], inputs.dtype)
            brainstate.nn.init_all_states(self.model)
            model.compile_graph(inp)
            model.show_graph()

        init()
        model = brainstate.nn.Vmap(model, vmap_states='new')

        def _etrace_grad(inp):
            out = model(inp)
            loss = self.f_loss(out, targets)
            return loss, out

        def _etrace_step(prev_grads, x):
            # no need to return weights and states, since they are generated then no longer needed
            cur_grads, local_loss, out = brainstate.augment.grad(
                _etrace_grad,
                self.weights,
                has_aux=True,
                return_value=True
            )(x)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, (out, local_loss)

        def _etrace_train(inputs_):
            grads = jax.tree.map(lambda a: jnp.zeros_like(a), self.weights.to_dict_values())
            grads, (outs, losses) = brainstate.compile.scan(_etrace_step, grads, inputs_)
            self.optimizer.update(grads)
            return losses.mean(), outs.sum(axis=0)

        # accuracy and loss
        loss, out_sum = _etrace_train(inputs)
        acc1, acc5 = accuracy(out_sum, labels, topk=(1, 5))

        return acc1, acc5, loss

    @brainstate.compile.jit(static_argnums=0)
    def predict(self, inputs, targets, labels):
        model = brainstate.nn.EnvironContext(self.model, fit=False)
        brainstate.nn.vmap_init_all_states(model, state_tag='hidden', axis_size=inputs.shape[1])
        model = brainstate.nn.Vmap(model, vmap_states='hidden')
        outs = brainstate.transform.for_loop(model, inputs)
        outs = outs.sum(axis=0)
        loss = self.f_loss(outs, targets)
        acc1, acc5 = accuracy(outs, labels, topk=(1, 5))
        return acc1, acc5, loss

    def do_epoch(self, do_training: bool, loader, bench_type: str, epoch: int):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        if bench_type == 'train':
            progress = ProgressMeter(len(loader), [batch_time, data_time, losses, top1, top5],
                                     prefix="Epoch: [{}]".format(epoch))
        else:
            progress = ProgressMeter(len(loader), [batch_time, losses, top1, top5], prefix='Test: ')

        end = time.time()
        for batch_idx, (data, label) in enumerate(loader):
            data_time.update(time.time() - end)
            data, label, target, timesteps = data_resizing(self.args, data, label)

            if do_training:
                if self.args.method == 'bptt':
                    acc1, acc5, loss = self.bptt_train(data, label, target)
                else:
                    acc1, acc5, loss = self.online_train(data, label, target)
            else:
                acc1, acc5, loss = self.predict(data, label, target)


            # measure accuracy and record loss
            n_batch = data.shape[1]
            losses.update(loss.item(), n_batch)
            top1.update(acc1, n_batch)
            top5.update(acc5, n_batch)

            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % self.args.print_freq == (self.args.print_freq - 1):
                progress.display(batch_idx)

        if bench_type == 'train':
            logging.info(' @Training * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        else:
            logging.info(' @Testing * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        return top1.avg, losses.avg

    def f_train(self, train_loader, test_loader):
        # Training and performance monitoring
        logging.info("\n=== Starting model training with %d epochs:\n" % (self.args.epochs,))
        best_acc1 = 0
        acc_train_hist = []
        acc_val_hist = []
        loss_train_hist = []
        loss_val_hist = []
        for epoch in range(1, self.args.epochs + 1):
            logging.info("\t Epoch " + str(epoch) + "...")
            # Will display the average accuracy on the training set during the epoch (changing weights)
            acc_t, loss_t = self.do_epoch(True, train_loader, 'train', epoch)
            acc_train_hist.append(float(acc_t))
            loss_train_hist.append(float(loss_t))
            # Check performance on the training set and on the test set:
            if not self.args.skip_test:
                acc1, loss_val = self.do_epoch(False, test_loader, 'test', epoch)
                acc_val_hist.append(float(acc1))
                loss_val_hist.append(float(loss_val))
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
                logging.info(f'Best acc at epoch {epoch}: {best_acc1}')
                if is_best:
                    state = {
                        'epoch': epoch,
                        'state_dict': self.weights.to_nest(),
                        'best_acc1': best_acc1,
                    }
                    braintools.file.msgpack_save(os.path.join(self.args.save_path, 'trial_model_best.msgpack'), state)
        np.save(self.args.save_path + f'/trial_train_acc.npy', np.array(acc_train_hist))
        np.save(self.args.save_path + f'/trial_val_acc.npy', np.array(acc_val_hist))
        np.save(self.args.save_path + f'/trial_train_loss.npy', np.array(loss_train_hist))
        np.save(self.args.save_path + f'/trial_val_loss.npy', np.array(loss_val_hist))


def data_resizing(args, data, label):
    timesteps = data.size(1)
    batch_size = data.size(0)
    data = data.float()
    data = data.view(batch_size, timesteps, 2, 32, 32)
    data = data.permute(1, 0, 3, 4, 2)
    if args.label_encoding == "one-hot":  # Do a one-hot encoding for classification
        target = brainstate.functional.one_hot(label, num_classes=args.n_classes)
    else:
        target = label
    label = label.view(-1)

    data = jnp.asarray(data)  # [n_seq, n_batch, height, width, n_channel]
    target = jnp.asarray(target)
    label = jnp.asarray(label, dtype=jnp.int32)

    return data, label, target, timesteps

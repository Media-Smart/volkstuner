import matplotlib
matplotlib.use('Agg')

import argparse, time, logging
import addict
import json

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms


def test(net, ctx, val_data):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
    return metric.get()


def _trainval(net, ctx, train_data, val_data, lr_scheduler, optimizer, batch_size, save_dir, save_period, reporter):
    # lr_scheduler
    num_epochs = lr_scheduler.num_epochs
    lr_decay = lr_scheduler.lr_decay
    lr_decay_epoch = [int(i) for i in lr_scheduler.lr_decay_epoch.split(',')] + [np.inf]

    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.Xavier(), ctx=ctx)

    # optimizer
    trainer = gluon.Trainer(net.collect_params(), optimizer.name,
                            {'learning_rate': optimizer.lr, 'wd': optimizer.wd, 'momentum': optimizer.momentum})

    metric = mx.metric.Accuracy()
    train_metric = mx.metric.Accuracy()
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    train_history = TrainingHistory(['training-error', 'validation-error'])

    iteration = 0
    lr_decay_count = 0

    best_val_score = 0

    for epoch in range(num_epochs):
        tic = time.time()
        train_metric.reset()
        metric.reset()
        train_loss = 0
        num_batch = len(train_data)
        alpha = 1

        if epoch == lr_decay_epoch[lr_decay_count]:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count += 1

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

            with ag.record():
                output = [net(X) for X in data]
                loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
            for l in loss:
                l.backward()
            trainer.step(batch_size)
            train_loss += sum([l.sum().asscalar() for l in loss])

            train_metric.update(label, output)
            name, acc = train_metric.get()
            iteration += 1

        train_loss /= batch_size * num_batch
        name, acc = train_metric.get()
        name, val_acc = test(net, ctx, val_data)
        reporter(epoch=epoch, accuracy=val_acc, train_acc=acc)
        train_history.update([1-acc, 1-val_acc])
        train_history.plot(save_path='%s/train_history.png'%save_dir)

        if val_acc > best_val_score:
            best_val_score = val_acc
            #net.save_parameters('%s/%.4f-cifar-%s-%d-best.params'%(save_dir, best_val_score, args.model, epoch))

        #logging.info('[Epoch %d] train=%f val=%f loss=%f time: %f' %
        #    (epoch, acc, val_acc, train_loss, time.time()-tic))

        if save_period and save_dir and (epoch + 1) % save_period == 0:
            net.save_parameters('%s/epoch-%d.params'%(save_dir, epoch))

    if save_period and save_dir:
        net.save_parameters('%s/epoch-%d.params'%(save_dir, num_epochs-1))



def trainval(args, reporter):
    #print('debug', args, type(args))

    classes = 10
    # runtime
    runtime = args.runtime
    batch_size = runtime.batch_size
    save_dir = runtime.save_dir
    save_period = runtime.save_period
    num_gpus = runtime.num_gpus
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = runtime.num_workers

    # model
    model_name = args.model.name
    if model_name.startswith('cifar_wideresnet'):
        kwargs = {'classes': classes,
                'drop_rate': args.model.drop_rate}
    else:
        kwargs = {'classes': classes}
    net = get_model(model_name, **kwargs)
    if args.model.resume_from:
        net.load_parameters(args.resume_from, ctx = context)

    save_period = runtime.save_period
    save_dir = runtime.save_dir + '/task_%d' % args.task_id
    makedirs(save_dir)
    args_fp = '%s/args.json' % save_dir
    with open(args_fp, 'w') as fd:
        json.dump(args, fd)

    transform_train = transforms.Compose([
        #gcv_transforms.RandomCrop(32, pad=4),
        #transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    train_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

    val_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if runtime.mode == 'hybrid':
        net.hybridize()
    #train(args.num_epochs, context)
    lr_scheduler = args.lr_scheduler
    optimizer = args.optimizer
    _trainval(net, context, train_data, val_data, lr_scheduler, optimizer, batch_size, save_dir, save_period, reporter)

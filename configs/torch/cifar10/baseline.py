import volkstuner.engine as ag


debug = True
num_epochs=200

trainval = dict(
    name='torch_cifar10',
    batch_size=128,
    num_workers=2,
    num_epochs=num_epochs,
    lr=ag.space.Real(1e-1, 1.1e-1, log=True),
    momentum=0.9,
    wd=ag.space.Real(1e-4, 1.1e-4, log=True),
    milestones=[100, 150],)

tuner = dict(
    num_gpus=1,
    num_cpus=4,
    epochs=num_epochs,
    num_trials=10,
    scheduler='fifo',
    checkpoint='checkpoint/cifar1.ag',)

import volkstuner.engine as ag


debug = False

# reuse
num_epochs=3
num_gpus = 1
trainval_name = 'mxnet_cifar10'
save_dir = 'exp/%s' % trainval_name

trainval = dict(
    name=trainval_name,
    model=dict(
        name='cifar_resnet20_v1',
        drop_rate=0,
        resume_from=None,
        ),
    optimizer=ag.Dict(
        name='nag',
        lr=ag.space.Real(1e-2, 1e-1, log=True), # 1e-1
        momentum=0.9,
        wd=ag.space.Real(1e-5, 1e-3, log=True), # 1e-4
        ),
    lr_scheduler=dict(
        lr_decay=0.1,
        lr_decay_epoch='100,150', 
        num_epochs=num_epochs,
        ),
    runtime=dict(
        batch_size=64,
        num_gpus=num_gpus,
        num_workers=2,
        save_period=1,
        save_dir=save_dir,
        mode='hybrid',
        ),
    )

tuner = dict(
    num_gpus=num_gpus,
    num_cpus=4,
    epochs=num_epochs,
    num_trials=3,
    scheduler='fifo',
    checkpoint='checkpoint/cifar1.ag',
    )

logger = dict(
    handlers=(
        dict(type='StreamHandler', level='DEBUG'),
        #dict(type='FileHandler', level='INFO'),
    ),
	save_dir=save_dir,
) 

import volkstuner.engine as ag


#debug = True
num_epochs=200
num_gpus = 1
job_name = 'torch_cifar10'
save_dir = 'workdir/%s' % job_name

job = dict(
    name=job_name,
    model=dict(
        name='mobilenet_v2',
        resume_from=None,
        ),
    optimizer=ag.Dict(
        lr=ag.space.Real(1e-1, 1.1e-1, log=True), # 1e-1
        momentum=0.9,
        wd=ag.space.Real(1e-4, 1.1e-4, log=True), # 1e-4
        ),
    lr_scheduler=dict(
        milestones=[100, 150],
        num_epochs=num_epochs,
        ),
    runtime=dict(
        batch_size=128,
        num_gpus=num_gpus,
        num_workers=8,
        save_period=50,
        save_dir=save_dir,
        ),
    )

tuner = dict(
    num_gpus=num_gpus,
    num_cpus=4,
    epochs=num_epochs,
    num_trials=10,
    scheduler='fifo',
    checkpoint='%s/tuner.ag' % save_dir,
    )

logger = dict(
    handlers=(
        dict(type='StreamHandler', level='DEBUG'),
        #dict(type='FileHandler', level='INFO'),
    ),
	save_dir=save_dir,
) 

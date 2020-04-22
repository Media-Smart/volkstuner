import os
import argparse
import logging
import sys
sys.path.insert(0, '.')
import numpy as np


import volkstuner.engine as ag
#import autogluon as ag
#print(ag.__file__)


from volkstuner.trainvals import get_trainval
from volkstuner.utils import Config
from volkstuner.utils import build_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='LEGO')
    parser.add_argument('config', help='Training config file path')
    args = parser.parse_args()
    return args


def main(cfg):
    args = cfg['tuner']

    build_logger(cfg['logger'])

    #if cfg['debug']:
    #    logging.basicConfig(level=logging.DEBUG)
    #else:
    #    logging.basicConfig(level=logging.INFO)

    trainval = get_trainval(cfg['trainval'])

    # create searcher and scheduler
    extra_node_ips = []
    if args['scheduler'] == 'hyperband':
        myscheduler = ag.scheduler.HyperbandScheduler(trainval,
                                                      resource={'num_cpus': args['num_cpus'], 'num_gpus': args['num_gpus']},
                                                      num_trials=args['num_trials'],
                                                      checkpoint=args['checkpoint'],
                                                      time_attr='epoch', reward_attr='accuracy',
                                                      max_t=args['epochs'], grace_period=args['epochs']//4,
                                                      dist_ip_addrs=extra_node_ips)
    elif args['scheduler'] == 'fifo':
        myscheduler = ag.scheduler.FIFOScheduler(trainval,
                                                 resource={'num_cpus': args['num_cpus'], 'num_gpus': args['num_gpus']},
                                                 num_trials=args['num_trials'],
                                                 checkpoint=args['checkpoint'],
                                                 reward_attr='accuracy',
                                                 dist_ip_addrs=extra_node_ips)
    else:
        raise RuntimeError('Unsuported Scheduler!')

    myscheduler.run()
    myscheduler.join_jobs()
    #myscheduler.get_training_curves('{}.png'.format(os.path.splitext(args['checkpoint'])[0]))
    logging.info('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
                                                                   myscheduler.get_best_reward()))

if __name__ == '__main__':
    args = parse_args()
    cfg_fp = args.config
    cfg = Config.fromfile(cfg_fp)
    main(cfg)

#!/usr/bin/env python
"""
This script runs a policy gradient algorithm
"""

from gym.envs import make
from modular_rl import *
import argparse, sys, cPickle
from tabulate import tabulate
import shutil, os, logging
import gym
import h5py
import boto3
import netifaces as ni
import copy

if __name__ == "__main__":
    s3_client = boto3.client('s3')

    ifs = ni.interfaces()
    ip_address = str(np.random.randint(0, 10000))
    for interface in ifs:
        if(interface[0:4] == 'eth0'):
            ip_address = ni.ifaddresses(interface)[2][0]['addr']
            break
    print ip_address

    f = file('logfile_' + ip_address + '.txt', 'w')
    # sys.stdout = f

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])

    # args.env = 'ObstacleGameMultiple-v0'
    # args.env = 'ObstacleGame3d-v0'
    args.env = 'RaceGame3d-v0'
    args.agent = 'modular_rl.agentzoo.TrpoAgent'
    timestep_limit = 5000
    no_timesteps = 5000
    # video_every = 1
    upload_to_s3_every = 20
    snapshot_every = 20
    no_iter = 1000000

    env = make(args.env)
    env_spec = env.spec
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    args.n_iter = no_iter
    args.timestep_limit = timestep_limit
    args.timesteps_per_batch = no_timesteps
    if args.timestep_limit == 0:
        args.timestep_limit = env_spec.timestep_limit
    args.outfile = '/tmp/' + ip_address
    hdf, diagnostics = prepare_h5_file(args)
    cfg = args.__dict__
    # np.random.seed(args.seed)
    agent = agent_ctor(env.observation_space, env.action_space, cfg)
    # h5f = h5py.File('/home/ubuntu/h5_172.31.0.171_620_108.a', 'r')
    # agent = cPickle.loads(h5f['agent_last_snapshot'].value)

    # hdf = h5py.File('/tmp/h5_172.31.0.171_620_36.a', "a")
    # hdf['agent_last_snapshot'] = np.array(cPickle.dumps(agent, -1))
    # hdf.flush()
    # hdf.close()
    # h3232[-1] = 0

    gym.logger.setLevel(logging.WARN)

    COUNTER = 0
    def callback(stats):
        global COUNTER
        COUNTER += 1
        # Print stats
        print "*********** Iteration %i ****************" % COUNTER
        print tabulate(filter(lambda (k,v) : np.asarray(v).size == 1, stats.items()))
        # if (COUNTER % upload_to_s3_every == 0) or (COUNTER == args.n_iter):
        #     s3_client.upload_file('logging_file.txt', 'machine-learning-bucket', 'logging_file_' + str(ip_address) + '.txt')
        #     s3_client.upload_file(args.outfile, 'machine-learning-bucket',
        #                       'h5_' + str(ip_address) + '.a')
        if (COUNTER % snapshot_every == 0) or (COUNTER == args.n_iter):
            snapshot_filename = 'h5_' + ip_address + "_" + str(COUNTER) + ".a"
            hdf = h5py.File('/tmp/' + snapshot_filename, "a")
            hdf['agent_last_snapshot'] = np.array(cPickle.dumps(agent, -1))
            hdf.flush()
            hdf.close()
            s3_client.upload_file('logfile_' + ip_address + '.txt', 'machine-learning-bucket',
                                  'logfile_' + ip_address + '.txt')
            s3_client.upload_file('/tmp/' + snapshot_filename, 'machine-learning-bucket', snapshot_filename)


    run_policy_gradient_algorithm(env, agent, callback=callback, usercfg = cfg)

import h5py
from modular_rl import *
from gym.envs import make
import cPickle
import random
from modular_rl import *

# Record pygame video by doing a rollout on the environment (with actions that do nothing at all)
random.seed(30000)
env = make('RaceGame3d-v0')

# WATCH OUT, this agent is not actually used when plotting, it's the one that is in ObstacleGameMultiple constructor

h5f = h5py.File('/Users/simon/Desktop/h5_macbook_36dist.a', 'r')
# h5f = h5py.File('/home/ubuntu/h5_172.31.37.149_100.a', 'r')
# h5f = h5py.File('/tmp/h5_dist.a', 'r')

agent = cPickle.loads(h5f['agent_last_snapshot'].value)

# env.monitor.start('video', video_callable=None, force=True)

do_rollouts_serial(env, agent, 1000, 999, itertools.count())

# env.monitor.close()
h5f.close()
"""
In this codebase, the "Agent" is a container with the policy, value function, etc.
This file contains a bunch of agents
"""


from modular_rl import *
from gym.spaces import Box, Discrete
from collections import OrderedDict
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers import Convolution2D, Flatten, Reshape, Input, merge
from keras.layers.advanced_activations import LeakyReLU
from modular_rl.trpo import TrpoUpdater
from modular_rl.ppo import PpoLbfgsUpdater, PpoSgdUpdater

MLP_OPTIONS = [
    ("hid_sizes", comma_sep_ints, [64, 64], "Sizes of hidden layers of MLP"),
    ("activation", str, "tanh", "nonlinearity")
]

def make_mlps(ob_space, ac_space, cfg):
    assert isinstance(ob_space, Box)
    hid_sizes = cfg["hid_sizes"]
    if isinstance(ac_space, Box):
        outdim = ac_space.shape[0]
        probtype = DiagGauss(outdim)
    elif isinstance(ac_space, Discrete):
        outdim = ac_space.n
        probtype = Categorical(outdim)
    net = Sequential()
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=ob_space.shape) if i==0 else {}
        net.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))
    if isinstance(ac_space, Box):
        net.add(Dense(outdim))
        Wlast = net.layers[-1].W
        Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
        net.add(ConcatFixedStd())
    else:
        net.add(Dense(outdim, activation="softmax"))
        Wlast = net.layers[-1].W
        Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
    policy = StochPolicyKeras(net, probtype)
    vfnet = Sequential()
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=(ob_space.shape[0]+1,)) if i==0 else {} # add one extra feature for timestep
        vfnet.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))
    vfnet.add(Dense(1))
    baseline = NnVf(vfnet, cfg["timestep_limit"], dict(mixfrac=0.1))
    return policy, baseline


def make_mlps_conv(ob_space, ac_space, cfg):
    outdim = ac_space.n
    probtype = Categorical(outdim)
    net = Sequential()
    img_width = int(np.sqrt(ob_space.shape[0]/4))
    img_shape = (img_width, img_width, 4)
    # img_width = int(np.sqrt(ob_space.shape[0]))
    # img_shape = (img_width, img_width, 1)
    net.add(Reshape(img_shape, input_shape=ob_space.shape))
    net.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='valid', activation='tanh', input_shape=img_shape))
    net.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='valid', activation='tanh'))
    net.add(Convolution2D(64, 3, 3, border_mode='valid', activation='tanh'))
    net.add(Flatten())
    net.add(Dense(512, activation='tanh'))
    net.add(Dense(outdim, activation="softmax"))
    Wlast = net.layers[-1].W
    Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
    policy = StochPolicyKeras(net, probtype)
    vfnet = Sequential()
    hid_sizes = cfg["hid_sizes"]
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=(ob_space.shape[0]+1,)) if i == 0 else {}  # add one extra feature for timestep
        # inshp = dict(input_shape=(ob_space.shape[0]*ob_space.shape[1]+1,)) if i==0 else {} # add one extra feature for timestep
        vfnet.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))
    vfnet.add(Dense(1))
    baseline = NnVf(vfnet, cfg["timestep_limit"], dict(mixfrac=0.1))
    return policy, baseline

def make_mlps_conv_2(ob_space, ac_space, cfg):
    outdim = ac_space.n
    probtype = Categorical(outdim)
    net = Sequential()
    img_width = int(np.sqrt((ob_space.shape[0]-1)/4))
    img_shape = (img_width, img_width, 4)

    input_img = Input(shape=(ob_space.shape[0]-1,))
    reshaped_input_img = Reshape(img_shape, input_shape=(ob_space.shape[0]-1,))(input_img)
    conv1 = Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='valid', activation='tanh', input_shape=img_shape)(reshaped_input_img)
    conv2 = Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='valid', activation='tanh')(conv1)
    conv3 = Convolution2D(64, 3, 3, border_mode='valid', activation='tanh')(conv2)
    flattened_conv3 = Flatten()(conv3)
    input_fc = Input(shape=(1,))
    concatenated = merge([flattened_conv3, input_fc], mode='concat')
    fc1 = Dense(512, activation='tanh')(concatenated)
    fc2 = Dense(outdim, activation="softmax")(fc1)
    net = Model(input=[input_img, input_fc], output=fc2)

    Wlast = net.layers[-1].W
    Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
    policy = StochPolicyKeras(net, probtype)
    vfnet = Sequential()
    hid_sizes = cfg["hid_sizes"]
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=(ob_space.shape[0]+1,)) if i == 0 else {}  # add one extra feature for timestep
        # inshp = dict(input_shape=(ob_space.shape[0]*ob_space.shape[1]+1,)) if i==0 else {} # add one extra feature for timestep
        vfnet.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))
    vfnet.add(Dense(1))
    baseline = NnVf(vfnet, cfg["timestep_limit"], dict(mixfrac=0.1))
    return policy, baseline

def make_mlps_conv_3(ob_space, ac_space, cfg):
    outdim = ac_space.n
    probtype = Categorical(outdim)
    net = Sequential()
    img_width = int(np.sqrt((ob_space.shape[0]-1)/1))
    # img_width = int(np.sqrt((ob_space.shape[0]-1-60)))
    img_shape = (img_width, img_width, 1)

    input_img = Input(shape=(ob_space.shape[0]-1,))
    reshaped_input_img = Reshape(img_shape, input_shape=(ob_space.shape[0]-1,))(input_img)
    conv1 = Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='valid', activation='tanh', input_shape=img_shape)(reshaped_input_img)
    conv2 = Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='valid', activation='tanh')(conv1)
    conv3 = Convolution2D(64, 3, 3, border_mode='valid', activation='tanh')(conv2)
    flattened_conv3 = Flatten()(conv3)
    input_fc = Input(shape=(1,))
    concatenated = merge([flattened_conv3, input_fc], mode='concat')
    fc1 = Dense(512, activation='tanh')(concatenated)
    # fc2 = Dense(64, activation='tanh')(fc1)
    fc2 = Dense(outdim, activation="softmax")(fc1)
    net = Model(input=[input_img, input_fc], output=fc2)

    Wlast = net.layers[-1].W
    Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
    policy = StochPolicyKeras(net, probtype)
    vfnet = Sequential()
    hid_sizes = cfg["hid_sizes"]
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=(ob_space.shape[0]+1,)) if i == 0 else {}  # add one extra feature for timestep
        # inshp = dict(input_shape=(ob_space.shape[0]*ob_space.shape[1]+1,)) if i==0 else {} # add one extra feature for timestep
        vfnet.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))
    vfnet.add(Dense(1))

    baseline = NnVf(vfnet, cfg["timestep_limit"], dict(mixfrac=0.1))
    return policy, baseline

def make_mlps_conv_4(ob_space, ac_space, cfg):
    outdim = ac_space.n
    probtype = Categorical(outdim)
    net = Sequential()

    no_sensors = 0

    img_width = int(np.sqrt((ob_space.shape[0]-no_sensors-1)/1))
    img_shape = (img_width, img_width, 1)

    input_img = Input(shape=(ob_space.shape[0]-no_sensors-1,))
    reshaped_input_img = Reshape(img_shape, input_shape=(ob_space.shape[0]-1-no_sensors,))(input_img)
    conv1 = Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='valid', activation='tanh', input_shape=img_shape)(reshaped_input_img)
    conv2 = Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='valid', activation='tanh')(conv1)
    conv3 = Convolution2D(64, 3, 3, border_mode='valid', activation='tanh')(conv2)
    flattened_conv3 = Flatten()(conv3)
    input_fc = Input(shape=(1+no_sensors,))
    concatenated = merge([flattened_conv3, input_fc], mode='concat')
    fc1 = Dense(512, activation='tanh')(concatenated)
    fc2 = Dense(outdim, activation="softmax")(fc1)
    net = Model(input=[input_img, input_fc], output=fc2)

    Wlast = net.layers[-1].W
    Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
    policy = StochPolicyKeras(net, probtype)
    vfnet = Sequential()
    hid_sizes = cfg["hid_sizes"]
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=(ob_space.shape[0]+1,)) if i == 0 else {}  # add one extra feature for timestep
        # inshp = dict(input_shape=(ob_space.shape[0]*ob_space.shape[1]+1,)) if i==0 else {} # add one extra feature for timestep
        vfnet.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))
    vfnet.add(Dense(1))

    baseline = NnVf(vfnet, cfg["timestep_limit"], dict(mixfrac=0.1))
    return policy, baseline

def make_mlps_conv_5(ob_space, ac_space, cfg):
    outdim = ac_space.n
    probtype = Categorical(outdim)
    net = Sequential()

    no_sensors = 27*4

    img_width = int(np.sqrt((ob_space.shape[0]-no_sensors-1)/1))
    # img_width = int(np.sqrt((ob_space.shape[0]-1-60)))
    img_shape = (img_width, img_width, 1)

    input_img = Input(shape=(ob_space.shape[0]-no_sensors-1,))
    reshaped_input_img = Reshape(img_shape, input_shape=(ob_space.shape[0]-1-no_sensors,))(input_img)
    conv1 = Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='valid', activation='tanh', input_shape=img_shape)(reshaped_input_img)
    conv2 = Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='valid', activation='tanh')(conv1)
    conv3 = Convolution2D(64, 3, 3, border_mode='valid', activation='tanh')(conv2)
    flattened_conv3 = Flatten()(conv3)
    input_fc = Input(shape=(1+no_sensors,))
    concatenated = merge([flattened_conv3, input_fc], mode='concat')
    fc1 = Dense(512, activation='tanh')(concatenated)
    # fc2 = Dense(64, activation='tanh')(fc1)
    fc2 = Dense(outdim, activation="softmax")(fc1)
    net = Model(input=[input_img, input_fc], output=fc2)

    # Agent trained without distance sensor inputs
    # h5f = h5py.File('/home/ubuntu/h5_172.31.42.192_340.a', 'r')
    h5f = h5py.File('/home/ubuntu/h5_172.31.0.171_620.a', 'r')
    agent_stored = cPickle.loads(h5f['agent_last_snapshot'].value)
    policy_net_stored = agent_stored.policy.net

    # Copy weights of the convolutional layer and second dense layer
    layer_conv1_stored = policy_net_stored.get_layer('convolution2d_1').get_weights()
    net.get_layer('convolution2d_1').set_weights(layer_conv1_stored)
    layer_conv2_stored = policy_net_stored.get_layer('convolution2d_2').get_weights()
    net.get_layer('convolution2d_2').set_weights(layer_conv2_stored)
    layer_conv3_stored = policy_net_stored.get_layer('convolution2d_3').get_weights()
    net.get_layer('convolution2d_3').set_weights(layer_conv3_stored)
    layer_d2_stored = policy_net_stored.get_layer('dense_2').get_weights()
    net.get_layer('dense_2').set_weights(layer_d2_stored)

    # Copy and pad the first dense layer weights with zeros for the new distance sensor input nodes
    layer_d1_stored = policy_net_stored.get_layer('dense_1').get_weights()
    layer_d1_stored_padded = layer_d1_stored
    layer_d1_stored_padded[0] = np.lib.pad(layer_d1_stored_padded[0], ((0, no_sensors), (0, 0)), 'constant', constant_values=0)
    net.get_layer('dense_1').set_weights(layer_d1_stored_padded)

    # Wlast = net.layers[-1].W
    # Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
    policy = StochPolicyKeras(net, probtype)
    vfnet = Sequential()
    hid_sizes = cfg["hid_sizes"]
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=(ob_space.shape[0]+1,)) if i == 0 else {}  # add one extra feature for timestep
        # inshp = dict(input_shape=(ob_space.shape[0]*ob_space.shape[1]+1,)) if i==0 else {} # add one extra feature for timestep
        vfnet.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))
    vfnet.add(Dense(1))

    # Copy and pad the first dense layer weights with zeros for the new distance sensor input nodes. Copy other dense layer weights.
    vfnet_stored = agent_stored.baseline.reg.net
    layer_dense3_stored = vfnet_stored.get_layer('dense_3').get_weights()
    layer_dense3_stored_padded = layer_dense3_stored
    layer_dense3_stored_padded[0] = np.lib.pad(layer_dense3_stored_padded[0], ((0, no_sensors), (0, 0)), 'constant', constant_values=0)
    vfnet.get_layer('dense_3').set_weights(layer_dense3_stored_padded)
    layer_dense4_stored = vfnet_stored.get_layer('dense_4').get_weights()
    vfnet.get_layer('dense_4').set_weights(layer_dense4_stored)
    layer_dense5_stored = vfnet_stored.get_layer('dense_5').get_weights()
    vfnet.get_layer('dense_5').set_weights(layer_dense5_stored)

    baseline = NnVf(vfnet, cfg["timestep_limit"], dict(mixfrac=0.1))
    return policy, baseline


def make_deterministic_mlp(ob_space, ac_space, cfg):
    assert isinstance(ob_space, Box)
    hid_sizes = cfg["hid_sizes"]
    if isinstance(ac_space, Box):
        outdim = ac_space.shape[0]
        probtype = DiagGauss(outdim)
    elif isinstance(ac_space, Discrete):
        outdim = ac_space.n
        probtype = Categorical(outdim)
    net = Sequential()
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=ob_space.shape) if i==0 else {}
        net.add(Dense(layeroutsize, activation="tanh", **inshp))
    inshp = dict(input_shape=ob_space.shape) if len(hid_sizes) == 0 else {}
    net.add(Dense(outdim, **inshp))
    Wlast = net.layers[-1].W
    Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
    policy = StochPolicyKeras(net, probtype)
    return policy

FILTER_OPTIONS = [
    ("filter", int, 1, "Whether to do a running average filter of the incoming observations and rewards")
]

def make_filters(cfg, ob_space):
    if cfg["filter"]:
        obfilter = ZFilter(ob_space.shape, clip=5)
        rewfilter = ZFilter((), demean=False, clip=10)
    else:
        obfilter = IDENTITY
        rewfilter = IDENTITY
    return obfilter, rewfilter


class AgentWithPolicy(object):
    def __init__(self, policy, obfilter, rewfilter):
        self.policy = policy
        self.obfilter = obfilter
        self.rewfilter = rewfilter
        self.stochastic = True
    def set_stochastic(self, stochastic):
        self.stochastic = stochastic
    def act(self, ob_no):
        return self.policy.act(ob_no, stochastic = self.stochastic)
    def get_flat(self):
        return self.policy.get_flat()
    def set_from_flat(self, th):
        return self.policy.set_from_flat(th)
    def obfilt(self, ob):
        return self.obfilter(ob)
    def rewfilt(self, rew):
        return self.rewfilter(rew)

class DeterministicAgent(AgentWithPolicy):
    options = MLP_OPTIONS + FILTER_OPTIONS
    def __init__(self, ob_space, ac_space, usercfg):
        cfg = update_default_config(self.options, usercfg)
        policy = make_deterministic_mlp(ob_space, ac_space, cfg)
        obfilter, rewfilter = make_filters(cfg, ob_space)
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)
        self.set_stochastic(False)

class TrpoAgent(AgentWithPolicy):
    options = MLP_OPTIONS + PG_OPTIONS + TrpoUpdater.options + FILTER_OPTIONS

    def __init__(self, ob_space, ac_space, usercfg):
        cfg = update_default_config(self.options, usercfg)
        # policy, self.baseline = make_mlps(ob_space, ac_space, cfg)
        policy, self.baseline = make_mlps_conv_4(ob_space, ac_space, cfg)
        obfilter, rewfilter = make_filters(cfg, ob_space)
        self.updater = TrpoUpdater(policy, cfg)
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)

# class TrpoAgent(AgentWithPolicy):
#     options = MLP_OPTIONS + PG_OPTIONS + TrpoUpdater.options + FILTER_OPTIONS
#     def __init__(self, ob_space, ac_space, usercfg):
#         cfg = update_default_config(self.options, usercfg)
#         # policy, self.baseline = make_mlps(ob_space, ac_space, cfg)
#         policy, self.baseline = make_mlps_conv_5(ob_space, ac_space, cfg)
#         obfilter, rewfilter = make_filters(cfg, ob_space)
#
#         # h5f = h5py.File('/tmp/h5_7763_4.a', 'r')
#         h5f = h5py.File('/home/ubuntu/h5_172.31.0.171_620.a', 'r')
#         agent_stored = cPickle.loads(h5f['agent_last_snapshot'].value)
#
#         L = agent_stored.obfilter.rs._M.shape[0]
#
#         obfilter.rs._M[:L] = agent_stored.obfilter.rs._M
#         obfilter.rs._S[:L] = agent_stored.obfilter.rs._S
#
#         obfilter.rs._n = agent_stored.obfilter.rs._n
#         obfilter.rs._M[L:] = 100
#
#         rewfilter = agent_stored.rewfilter
#
#         self.updater = TrpoUpdater(policy, cfg)
#         AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)

class PpoLbfgsAgent(AgentWithPolicy):
    options = MLP_OPTIONS + PG_OPTIONS + PpoLbfgsUpdater.options + FILTER_OPTIONS
    def __init__(self, ob_space, ac_space, usercfg):
        cfg = update_default_config(self.options, usercfg)
        policy, self.baseline = make_mlps(ob_space, ac_space, cfg)
        obfilter, rewfilter = make_filters(cfg, ob_space)
        self.updater = PpoLbfgsUpdater(policy, cfg)
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)

class PpoSgdAgent(AgentWithPolicy):
    options = MLP_OPTIONS + PG_OPTIONS + PpoSgdUpdater.options + FILTER_OPTIONS
    def __init__(self, ob_space, ac_space, usercfg):
        cfg = update_default_config(self.options, usercfg)
        policy, self.baseline = make_mlps(ob_space, ac_space, cfg)
        obfilter, rewfilter = make_filters(cfg, ob_space)
        self.updater = PpoSgdUpdater(policy, cfg)
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)



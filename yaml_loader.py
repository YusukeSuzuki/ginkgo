import yaml
import tensorflow as tf

# ------------------------------------------------------------
# utilities
# ------------------------------------------------------------

def weight_variable(shape, dev=0.35, name=None):
    """create weight variable for conv2d(weight sharing)"""

    return tf.get_variable(name, shape,
        initializer=tf.truncated_normal_initializer(stddev=dev))

def bias_variable(shape, val=0.1, name=None):
    """create bias variable for conv2d(weight sharing)"""

    return tf.get_variable(
        name, shape, initializer=tf.constant_initializer(val))

class WithNone:
    def __enter__(self): pass
    def __exit__(self,t,v,tb): pass

# ------------------------------------------------------------
# assert
# ------------------------------------------------------------

def nop(val):
    return val

def is_exist(key, val):
    return (val, '{} required'.format(key))

def not_empty(key, val):
    return (len(val) > 0, '{} must not be empty'.format(key))

class is_greater_than:
    def __init__(self, val):
        self.val = val

    def __call__(self, key, val):
        return (val > self.val, '{} must be > {}'.format(key, self.val))

class is_typeof:
    def __init__(self, val):
        self.val = val

    def __call__(self, key, val):
        return (type(val) is self.val, '{} must be {}'.format(key, self.val))

# ------------------------------------------------------------
# YAML Graph Nodes
# ------------------------------------------------------------

class Loader(yaml.Loader):
    def __init__(self, stream):
        self.nids = {}
        super(Loader, self).__init__(stream)

class Node:
    def __init__():
        pass

    def parse(self,loader,node,params):
        yaml_dict = loader.construct_mapping(node, deep=True)
        self.__dict__['nid'] =  None

        for key, value in params.items():
            got = yaml_dict.get(key, value[1])
            self.__dict__[key] = value[0](got) if got is not None else got

        assert self.nid not in loader.nids, \
            'line {}: nid "{}" is already exists at line {}'.format(
                node.end_mark.line+1, self.nid, loader.nids[self.nid].line+1)

        for key, val in params.items():
            for cond in val[2]:
                ret, mess = cond(key, self.__dict__[key])
                assert ret, 'line {}: {}'.format(node.end_mark.line+1, mess)

        if self.nid: loader.nids[self.nid] = node.end_mark

    def build(self, nids, exclude_tags):
        for exclude_tag in exclude_tags:
            if exclude_tag in self.tags:
                return nids

        nids, nid, node = self.create_node(nids, exclude_tags)
 
        if nid and node is not None: nids[nid] = node

        return nids

class Root(yaml.YAMLObject):
    yaml_tag = u'!root'

    def __init__(self, nodes, nodes_required):
        self.nodes = nodes
        self.nodes_required = nodes_required

    def __repr__(self):
        return 'Root'

    @classmethod
    def from_yaml(cls, loader, node):
        yaml_dict = loader.construct_mapping(node)

        args = {
            'nodes': yaml_dict.get('nodes', None),
            'nodes_required': yaml_dict.get('nodes_required', None),
        }

        assert type(args['nodes']) is list, \
            'line {}: nodes must be list'.format(node.end_mark.line+1)
        assert type(args['nodes_required']) is list, \
            'line {}: nodes_required must be list'.format(node.end_mark.line+1)

        for required_node in args['nodes_required']:
            assert required_node not in loader.nids, \
                'line {}: nid "{}" is already exists'.format(node.end_mark.line+1, required_node)
            print(required_node)
            loader.nids[required_node] = node.end_mark

        return cls(**args)

    def build(self, feed_dict={}, exclude_tags=[]):
        self.__nids = {}

        for key, val in feed_dict.items():
            self.__nids[key]  = val

        for required_node in self.nodes_required:
            if required_node not in self.__nids:
                raise ValueError('feed_dict requires {}.'.format(self.nodes_required))

        for node in self.nodes:
            self.__nids = node.build(self.__nids, exclude_tags)

class With(yaml.YAMLObject, Node):
    yaml_tag = u'!with'

    def __init__(self, loader, node):
        params = {
            'tags': (nop, [], [is_typeof(list)]),
            'nodes': (nop, [], [is_typeof(list), not_empty]),
            'variable_scope': (str, '', []),
            'name_scope': (str, '', []),
            'device_scope': (str, '', [])
            }

        self.parse(loader, node, params)

        assert self.variable_scope or self.name_scope or self.device_scope, \
            'at leaset each of variable, name, device should not be None'

    def __repr__(self):
        return 'With'

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(loader, node)

    def create_node(self, nids, exclude_tags):
        vs = lambda x: ( tf.variable_scope(x) if x else WithNone() )
        ns = lambda x: ( tf.name_scope(x) if x else WithNone() )
        dv = lambda x: ( tf.device(x) if x else WithNone() )

        with vs(self.variable_scope), ns(self.name_scope), dv(self.device_scope):
            for node in self.nodes:
                nids = node.build(nids, exclude_tags)

        return nids, None, None

class Conv2d(yaml.YAMLObject, Node):
    yaml_tag = u'!conv2d'

    def __init__(self, loader, node):
        params = {
            'nid': (str, None, []),
            'tags': (nop, [], [is_typeof(list)]),
            'source': (nop, None, [is_exist]),
            'width': (int, 0, [is_greater_than(0)]),
            'height': (int, 0, [is_greater_than(0)]),
            'kernels_num': (int, 0, [is_greater_than(0)]),
            'strides': (nop, [1,1,1,1], [is_typeof(list)]),
            'b_init': (float, 0.1, []),
            'padding': (str, 'SAME', []),
            'name': (str, None, []),
            'variable_scope': (str, None, [])
            }
        self.parse(loader, node, params)

    def __repr__(self):
        return 'Conv2d'

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(loader, node)

    def create_node(self, nids, exclude_tags):
        source_node = nids[self.source]
        channels = source_node.get_shape()[3]

        with tf.variable_scope(self.variable_scope) if self.variable_scope else WithNone():
            w = weight_variable(
                [self.height, self.width, channels,self.kernels_num], name="weight")
            b = bias_variable([self.kernels_num], val=self.b_init, name="bias")

        return nids, self.nid, tf.add( tf.nn.conv2d(
                source_node, w, strides=self.strides, padding=self.padding), b, name=self.name)

class Conv2dTranspose(yaml.YAMLObject, Node):
    yaml_tag = u'!conv2d_transpose'

    def __init__(self, nid, source, shape_as, width, height, strides=[1,1,1,1],
        b_init=0.1, padding='SAME', name=None, variable_scope=None):

        self.nid = nid
        self.name = name
        self.variable_scope = variable_scope
        self.source = source
        self.shape_as = shape_as
        self.width = width
        self.height = height
        self.strides= strides
        self.padding = padding
        self.b_init = b_init

    def __init__(self, loader, node):
        params = {
            'nid': (str, None, []),
            'tags': (nop, [], [is_typeof(list)]),
            'source': (nop, None, [is_exist]),
            'shape_as': (nop, None, [is_exist]),
            'width': (int, 0, [is_greater_than(0)]),
            'height': (int, 0, [is_greater_than(0)]),
            'strides': (nop, [1,1,1,1], [is_typeof(list)]),
            'b_init': (float, 0.1, []),
            'padding': (str, 'SAME', []),
            'name': (str, None, []),
            'variable_scope': (str, None, [])
            }
        self.parse(loader, node, params)

    def __repr__(self):
        return 'Conv2dTranspose'

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(loader, node)

    def create_node(self, nids, exclude_tags):
        source_node = nids[self.source]
        shape_as = nids[self.shape_as]

        with tf.variable_scope(self.variable_scope) if self.variable_scope else WithNone():
            shape = source_node.get_shape()
            out_shape = shape_as.get_shape()
            w = tf.get_variable('weight',
                [self.height, self.width, out_shape[3], shape[3]],
                initializer=tf.truncated_normal_initializer(stddev=0.35))

            return nids, self.nid, tf.nn.conv2d_transpose(
                source_node, w, out_shape, strides=self.strides,
                padding=self.padding, name=self.name)

class Conv2dAELoss(yaml.YAMLObject, Node):
    yaml_tag = u'!conv2d_ae_loss'

    def __init__(self, loader, node):
        params = {
            'nid': (str, None, []),
            'tags': (nop, [], [is_typeof(list)]),
            'source1': (nop, None, [is_exist]),
            'source2': (nop, None, [is_exist]),
            'name': (str, None, []),
            'variable_scope': (str, None, [])
            }
        self.parse(loader, node, params)

    def __repr__(self):
        return 'Conv2dAELoss'

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(loader, node)

    def create_node(self, nids, exclude_tags):
        source1 = nids[self.source1]
        source2 = nids[self.source2]

        with tf.variable_scope(self.variable_scope) if self.variable_scope else WithNone():
            return nids, self.nid, tf.squared_difference(source1, source2, name=self.name)

class AdamOptimizer(yaml.YAMLObject, Node):
    yaml_tag = u'!adam_optimizer'

    def __init__(self, nid, source, val=1e-4, name=None):
        self.nid = nid
        self.name = name
        self.source = source
        self.val = val

    def __init__(self, loader, node):
        params = {
            'nid': (str, None, []),
            'tags': (nop, [], [is_typeof(list)]),
            'source': (nop, None, [is_exist]),
            'val': (float, 1e-4, []),
            'name': (str, None, []),
            }
        self.parse(loader, node, params)

    def __repr__(self):
        return 'AdamOptimizer'

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(loader, node)

    def create_node(self, nids, exclude_tags):
        source_node = nids[self.source]

        global_step = tf.get_variable(
            'global_step', (),
            initializer=tf.constant_initializer(0), trainable=False)
        return nids, self.nid, tf.train.AdamOptimizer(self.val).minimize(
            source_node, global_step=global_step, name=self.name)

class MaxPool2x2(yaml.YAMLObject, Node):
    yaml_tag = u'!max_pool_2x2'

    def __init__(self, loader, node):
        params = {
            'nid': (str, None, []),
            'tags': (nop, [], [is_typeof(list)]),
            'source': (nop, None, [is_exist]),
            'ksize': (nop, [1,2,2,1], [is_typeof(list)]),
            'strides': (nop, [1,2,2,1], [is_typeof(list)]),
            'padding': (str, 'SAME', []),
            'name': (str, None, []),
            }
        self.parse(loader, node, params)

    def __repr__(self):
        return 'MaxPool2x2'

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(loader, node)

    def create_node(self, nids, exclude_tags):
        source_node = nids[self.source]

        return nids, self.nid, tf.nn.max_pool(
            source_node, ksize=self.ksize, strides=self.strides,
            padding=self.padding, name=self.name)

class ReduceMean(yaml.YAMLObject, Node):
    yaml_tag = u'!reduce_mean'

    def __init__(self, loader, node):
        params = {
            'nid': (str, None, []),
            'tags': (nop, [], [is_typeof(list)]),
            'source': (nop, None, [is_exist]),
            'reduction_indices': (nop, None, []),
            'keep_dims': (bool, False, []),
            'name': (str, None, []),
            }
        self.parse(loader, node, params)

    def __repr__(self):
        return 'ReduceMean'

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(loader, node)

    def create_node(self, nids, exclude_tags):
        source_node = nids[self.source]

        return nids, self.nid, tf.reduce_mean(
            source_node, reduction_indices=self.reduction_indices,
            keep_dims=self.keep_dims, name=self.name)

class ScalarSummary(yaml.YAMLObject, Node):
    yaml_tag = u'!scalar_summary'

    def __init__(self, nid, summary_tag, source, name=None):

        self.nid = nid
        self.name = name
        self.summary_tag = summary_tag
        self.source = source

    def __init__(self, loader, node):
        params = {
            'nid': (str, None, []),
            'tags': (nop, [], [is_typeof(list)]),
            'source': (nop, None, [is_exist]),
            'summary_tag': (str, '', [not_empty]),
            'name': (str, None, []),
            }
        self.parse(loader, node, params)

    def __repr__(self):
        return 'ScalarSummary'

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(loader, node)

    def create_node(self, nids, exclude_tags):
        source_node = nids[self.source]

        return nids, self.nid, tf.scalar_summary(
            self.summary_tag, source_node, name=self.name)

class ImageSummary(yaml.YAMLObject, Node):
    yaml_tag = u'!image_summary'

    def __init__(self, loader, node):
        params = {
            'nid': (str, None, []),
            'tags': (nop, [], [is_typeof(list)]),
            'source': (nop, None, [is_exist]),
            'max_images': (int, 3, [is_greater_than(0)]),
            'summary_tag': (str, '', [not_empty]),
            'name': (str, None, []),
            }
        self.parse(loader, node, params)

    def __repr__(self):
        return 'ImageSummary'

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(loader, node)

    def create_node(self, nids, exclude_tags):
        source_node = nids[self.source]

        return nids, self.nid, tf.image_summary(
            self.summary_tag, source_node, max_images=self.max_images, name=self.name)

# ------------------------------------------------------------
# Loader function
# ------------------------------------------------------------

def load(path):
    graph= yaml.load(open(str(path)).read(), Loader=Loader)

    if type(graph['root']) is not Root:
        raise IOError("no Root in yaml file")

    return graph['root']


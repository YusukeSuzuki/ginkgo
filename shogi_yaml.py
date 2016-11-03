import tensorflow as tf
import yaml
import yaml_loader as yl

class ProphetLoss(yaml.YAMLObject, yl.Node):
    yaml_tag = u'!prophet_loss'

    def __init__(self, loader, node):
        params = {
            'nid': (str, None, []),
            'tags': (yl.nop, [], [yl.is_typeof(list)]),
            'source1': (yl.nop, None, [yl.is_exist]),
            'source2': (yl.nop, None, [yl.is_exist]),
            'source3': (yl.nop, None, [yl.is_exist]),
            'name': (str, None, []),
            'variable_scope': (str, None, [])
            }
        self.parse(loader, node, params)

    def __repr__(self):
        return 'ProphetLoss'

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(loader, node)

    def create_node(self, nids, exclude_tags):
        source_node1 = nids[self.source1]
        source_node2 = nids[self.source2]
        source_node3 = nids[self.source3]

        with tf.variable_scope(self.variable_scope) \
                if self.variable_scope else yl.WithNone():
            loss = tf.reduce_mean( -tf.reduce_sum( source_node2 * tf.log(source_node1), reduction_indices=[1] ) ) * source_node3
            return nids, self.nid, loss

class CorrectRate(yaml.YAMLObject, yl.Node):
    yaml_tag = u'!correct_rate'

    def __init__(self, loader, node):
        params = {
            'nid': (str, None, []),
            'tags': (yl.nop, [], [yl.is_typeof(list)]),
            'x1': (yl.nop, None, [yl.is_exist]),
            'x2': (yl.nop, None, [yl.is_exist]),
            'name': (str, None, []),
            'variable_scope': (str, None, [])
            }
        self.parse(loader, node, params)

    def __repr__(self):
        return 'CorrectRate'

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(loader, node)

    def create_node(self, nids, exclude_tags):
        x1 = nids[self.x1]
        x2 = nids[self.x2]

        with tf.variable_scope(self.variable_scope) \
                if self.variable_scope else yl.WithNone():

            corrects = tf.equal(tf.argmax(x1,1), tf.argmax(x2,1))
            rate = tf.reduce_mean(tf.cast(corrects, tf.float32))

            with tf.device('/cpu:0'):
                tf.scalar_summary('correct_rate/'+rate.device, rate)

            return nids, self.nid, rate


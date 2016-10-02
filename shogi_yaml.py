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
            loss = tf.squared_difference(source_node1, source_node2) * source_node3
            return nids, self.nid, loss


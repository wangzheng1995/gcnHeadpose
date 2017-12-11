import utils.buildGraph as bg
import tensorflow as tf
import os

from dataset import Dataset

os.environ['CUDA_VISIBLE_DEVICES']='0'

from models import GCN, MLP

model = 'GCN'

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn', 'model name')
flags.DEFINE_string('epochs', 2000, 'single batch')
flags.DEFINE_string('dropout', 0., 'dropout rate')
flags.DEFINE_string('weight_decay', 5e-4, 'weight_decay value')
flags.DEFINE_string('hidden1', 10, 'hidden1 dim')
flags.DEFINE_string('hidden2', 5, 'hidden2 dim')
flags.DEFINE_string('learning_rate', 0.001, 'Initial learning rate')

# Load data
#feat, lbl, adj = bg.prepare()
#feat = [bg.preprocess_feat(f) for f in feat]
ds = Dataset()
if model == 'GCN':
    #adj = [[bg.preprocess_adj(a)] for a in adj]
    #support = adj
    num_supports = 1
    model_func = GCN

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=(None, 5)),#tf.constant(feat[0][2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.int64, shape=(None)),#lbl[0].shape[0])),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Creat model
model = model_func(placeholders, input_dim=5, logging=True)
# Initialize session
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    return None

# Init variables
sess.run(tf.global_variables_initializer())

# Train model
for epoch in range(100):#FLAGS.epochs):
    # Construct feed dictionary
    #print ">>>>>>"
    #print feat[0]
    #print support[0]
    #print lbl[0]
    i = epoch
    bs = 1
    end = False
    loss = 1.
    accuracy = 0.
    count = 0
    while not end:
        end, feat, lbl, support = ds.sample_batch(bs)
        feed_dict = bg.construct_feed_dict(feat[0], support[0], lbl[0], placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.inputs, model.outputs], feed_dict=feed_dict)
        #print (epoch, outs[1], outs[2])
        loss = loss * .9 + outs[1] * .1
        accuracy = accuracy * .9 + outs[2] * .1
        if count % 20 ==0:
            print ("%d, %.3f, %.3f" % (epoch, loss, accuracy))
            count = 0
        count += 1
        #print "feat"
        #print outs[3]
        #print outs[4]
        #print lbl[0]
        
    #print (outs[2])
    #print (outs[4])

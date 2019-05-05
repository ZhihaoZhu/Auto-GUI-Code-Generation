import tensorflow as tf
import numpy as np
import os

global_step = tf.Variable(0, trainable=False)
increment_op = tf.assign_add(global_step, tf.constant(1))
lr = tf.train.exponential_decay(0.1, global_step, decay_steps=1, decay_rate=0.9, staircase=False)

tf.summary.scalar('learning_rate', lr)

sum_ops = tf.summary.merge_all()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

if not os.path.isdir("./tmp"):
    os.mkdir("./tmp")

summary_writer = tf.summary.FileWriter('./tmp/', sess.graph)

for step in range(0, 10):
    s_val = sess.run(sum_ops)
    summary_writer.add_summary(s_val, global_step=step)
    sess.run(increment_op)

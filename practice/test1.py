import tensorflow as tf

################graph
a = tf.constant([1.2, 2.3], name='a')
b = tf.constant([2.3, 4.], name="b")
result = a + b
print(a.graph is tf.get_default_graph())  #True
print(b.graph is tf.get_default_graph())  #True
print(result.graph is tf.get_default_graph())  #True

# 不同计算图上的张量和运算不会共享 通过tf.Graph来生成新的图

g1 = tf.Graph()
with g1.as_default():
    #在g1 中定义变量
    v = tf.get_variable("v", initializer=tf.zeros_initializer()(shape=[1]))

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", initializer=tf.ones_initializer()(shape=[1]))

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))


with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

# ###################TensorFlow随机生成的函数
# 正态分布 mean平均值 stddev标准差
a = tf.random_normal(shape=(2,3), mean=2, stddev=1, dtype=tf.float32)

# 如果随机出来的值偏离平均值超过两个标准差 会重新生成
b = tf.truncated_normal(shape=(3,), mean=1, stddev=0.1, dtype=tf.float32)

#均匀分布
c = tf.random_uniform(shape=(3, ), minval=3, maxval=5)

with tf.Session() as sess:
    print(sess.run([a, b, c]))

#通过设置assign(validate=false)可以改变变量的维度

########################完成到3.4.4待续...    date:6-24

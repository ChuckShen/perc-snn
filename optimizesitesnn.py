import tensorflow as tfv
import numpy as np
import random
import os
import time
start=time.time()

tf = tfv.compat.v1
tf.compat.v1.disable_eager_execution()
sigmoid = tf.keras.activations.hard_sigmoid

P1 = 0.1
P2 = 0.9
prob = 0.01
LENGTH = 40

data_path = '2D_percolation' + '/' + str(LENGTH) + '/'
files_list = sorted(os.listdir(data_path))

tf.reset_default_graph()
Format = '.npy'
learningrate = 1e-4
Epoch = 3000
support_set1 = [round(prob * i, 2) for i in range(0, int(P1 / prob) + 1)]
support_set1 = [f"{x:.2f}" for x in support_set1]
support_set2 = [round(prob * i, 2) for i in range(int(P2 / prob), int(1 / prob) + 1)]
support_set2 = [f"{x:.2f}" for x in support_set2]
print(support_set1)
print(support_set2)
support_set = support_set1 + support_set2

support_point = [round(prob * i, 2) for i in range(0, int(1 / prob) + 1)]
support_point = [support_point[0], support_point[-1], support_point[39], support_point[58], support_point[79]]
support_point = [f"{x:.2f}" for x in support_point]
print(support_point)

def Network(net):
    net = tf.layers.flatten(net)
    net = tf.layers.Dense(units=128, activation=None)(net)
    net = tf.nn.swish(tf.layers.BatchNormalization()(net))
    net = tf.layers.Dense(units=32, activation=None)(net)
    net = tf.nn.sigmoid(tf.layers.BatchNormalization()(net))
    return net

def Class(net):
    net = tf.layers.Dense(units=32, activation=None)(net)
    net = tf.nn.swish(tf.layers.BatchNormalization()(net))
    net = tf.layers.Dense(units=1, activation=None)(net)
    net = tf.nn.sigmoid(tf.layers.BatchNormalization()(net))
    return net

def data_generator(support_set1, support_set2, data_path):
    while True:
        list1 = random.choice([support_set1, support_set2])
        list2 = random.choice([support_set1, support_set2])
        P_train1 = random.choice(list1)
        P_train2 = random.choice(list2)
        data_P1 = np.load(data_path + str(P_train1) + Format)
        data_P2 = np.load(data_path + str(P_train2) + Format)
        if list2 == list1:
            label_train = np.ones([num_sample, 1])
            yield (data_P1, data_P2), label_train, 'positive'
        else:
            label_train = np.zeros([num_sample, 1])
            yield (data_P1, data_P2), label_train, 'negative'

config_point1 = np.load(data_path + str(support_point[0]) + Format)
shape = np.shape(config_point1)[1:]
num_sample = np.shape(config_point1)[0]

positive = np.ones([num_sample, 1])
negative = np.zeros([num_sample, 1])

Input2 = tf.placeholder(tf.float32, shape=(None,) + shape, name='input2')
Input1 = tf.placeholder(tf.float32, shape=(None,) + shape, name='input1')
label = tf.placeholder(tf.float32, shape=[None, 1], name='labels')

# 使用tf.data API创建数据集
dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(support_set1, support_set2, data_path),
    output_types=((tf.float32, tf.float32), tf.float32, tf.string),
    output_shapes=(((None,) + shape, (None,) + shape), (None, 1), ())
)

dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
iterator = dataset.make_one_shot_iterator()
((Input1, Input2), label, label_info) = iterator.get_next()

output1 = Network(Input1)
output2 = Network(Input2)
result = Class(tf.math.abs(output1 - output2))
loss = tf.reduce_mean(-tf.reduce_sum(label * tf.log(result + 1e-8) + (1 - label) * tf.log(1 - result + 1e-8), -1), 0)
solver = tf.train.AdamOptimizer(learningrate).minimize(loss)

# 配置会话使用GPU
sess = tf.Session(config=tf.ConfigProto(
    log_device_placement=False,
    gpu_options=tf.GPUOptions(allow_growth=True)
))

sess.run(tf.global_variables_initializer())
sess.graph.finalize()

for epoch in range(Epoch):
    try:
        _, loss_np, label_info_np = sess.run([solver, loss, label_info])
        print(f"Epoch {epoch}, Loss: {loss_np}, Label: {label_info_np.decode('utf-8')}")
    except tf.errors.OutOfRangeError:
        break

Result = []

for files in files_list:
    Final = []
    Final.append(float(files.split(".npy")[0]))
    for point in support_point:
        config_point = np.load(data_path + str(point) + Format)
        config = np.load(data_path + files)
        feed = {Input1: config, Input2: config_point}
        final = sess.run(result, feed)
        Final.append(np.mean(final))
    print(Final)
    Result.append(Final)

end=time.time()
print('Running time: %s Seconds'%(end-start))
np.savetxt(f'{LENGTH}_{P1}_{P2}_optimize.dat', Result)

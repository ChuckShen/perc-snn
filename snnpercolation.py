import tensorflow as tfv
import numpy as np
import random, os
import time
start=time.time()

tf = tfv.compat.v1
tf.compat.v1.disable_eager_execution()
sigmoid = tf.keras.activations.hard_sigmoid

P1 = 0.1
P2 = 0.9
prob = 0.01
LENGTH = 40

data_path = '2D_percolation'+'/' + str(LENGTH) +'/'
files_list = sorted(os.listdir(data_path))


tf.reset_default_graph()
Format = '.npy' 
learningrate = 1e-4
Epoch = 3000
support_set1 = [round(prob*i, 2) for i in range(0, int(P1/prob)+1)]
support_set1 = [f"{x:.2f}" for x in support_set1]
support_set2 = [round(prob*i, 2) for i in range(int(P2/prob), int(1/prob)+1)]
support_set2 = [f"{x:.2f}" for x in support_set2]
print(support_set1)
print(support_set2)
support_set = support_set1 + support_set2

# support_point1 = support_set1[0]
#support_point2 = support_set2[-1]
support_point = [round(prob*i, 2) for i in range(0, int(1/prob)+1)]
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
# swish
def Class(net):
    # net = tf.layers.BatchNormalization()(net)
    net = tf.layers.Dense(units=32, activation=None)(net)
    net = tf.nn.swish(tf.layers.BatchNormalization()(net))
    net = tf.layers.Dense(units=1, activation=None)(net)
    net = tf.nn.sigmoid(tf.layers.BatchNormalization()(net))
    return net

config_point1 = np.load(data_path+str(support_point[0])+Format)
#config_point2 = np.load(data_path+str(support_point2)+Format)
shape = np.shape(config_point1)[1:]
#TIME = np.shape(config_point)[-1]
num_sample = np.shape(config_point1)[0]
#result = tf.math.reduce_mean(network(config_point),0)


positive = np.ones([num_sample, 1])#[[1] for i in range(num_sample)]
# tf.ones([num_sample,1])
negative = np.zeros([num_sample, 1])# [[0] for i in range(num_sample)]
# tf.zeros([num_sample, 1])

Input2 = tf.placeholder(tf.float32, shape=(None,)+shape, name = 'input2')
Input1 = tf.placeholder(tf.float32, shape=(None,)+shape, name = 'input1')
label = tf.placeholder(tf.float32, shape=[None, 1], name = 'labels')


output1 = Network(Input1)
output2 = Network(Input2)
result = Class(tf.math.abs(output1-output2))
# result = Class(output1 - output2)
# loss = tf.reduce_mean(tf.reduce_sum((result-label)**2/0.001, -1), 0)
loss = tf.reduce_mean(- tf.reduce_sum( label*tf.log(result+1e-8) + (1-label)*tf.log(1-result+1e-8), -1), 0)
# Euclidean distance between embeddings
# distance = tf.norm(output1 - output2, axis=-1)

# Contrastive loss with margin
# margin = 1  # 调整此值根据需要
# contrastive_loss = (1 - label) * 0.5 * tf.square(result) + label * 0.5 * tf.square(tf.maximum(0.0, margin - result))
#
# # Average the loss
# loss = tf.reduce_mean(contrastive_loss)

solver = tf.train.AdamOptimizer(learningrate).minimize(loss)
sess = tf.Session(config=tf.ConfigProto(
                                        log_device_placement=False))

sess.run(tf.global_variables_initializer())
sess.graph.finalize()

for epoch in range(Epoch):
    list1 = random.choice([support_set1, support_set2])
    list2 = random.choice([support_set1, support_set2])
    P_train1 = random.choice(list1)
    P_train2 = random.choice(list2)
    print('epoch:', epoch)
    print(P_train1, P_train2)
    if list2 == list1:
        label_train = positive
        print('positive')
    else:
        label_train = negative
        print('negative')

    data_P1 = np.load(data_path+str(P_train1)+Format)
    data_P2 = np.load(data_path+str(P_train2)+Format)

    np.random.shuffle(data_P1)
    np.random.shuffle(data_P2)

    feed = {label:label_train, Input1:data_P1, Input2:data_P2}
    _, loss_np = sess.run([solver, loss], feed)
    print(loss_np)
    print()


Result = []

for files in files_list:
    Final = []
    # Final.append(float(files.split(".npy")[0]))
    Final.append(float(files.split(".npy")[0]))
    for point in support_point:
        config_point = np.load(data_path+str(point)+Format)
        config = np.load(data_path+files)
        feed = {Input1:config, Input2:config_point}
        final = sess.run(result, feed)
    #   feed2 = {Input1:config, Input2:config_point2}
    #   final2= sess.run(result, feed2)
        # print(files.split('.npy')[0])
        Final.append(np.mean(final))
    # print([float(files.split('.npy')[0]), np.mean(final1), np.mean(final2)])
    print(Final)
    Result.append(Final)
    # Result.append([float(files.split('.npy')[0]), np.mean(final1), np.mean(final2)])
end=time.time()
print('Running time: %s Seconds'%(end-start))
np.savetxt( str(LENGTH) +'_' +str(P1)+'_'+str(P2)+'.dat', Result)
# np.savetxt('2D_percolation/'+str(LENGTH)+'_'+str(P1)+'_'+str(P2) +'.dat', Result)
# + str(x) + '_'
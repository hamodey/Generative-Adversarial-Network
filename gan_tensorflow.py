import os, time, itertools, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# G(z)
def generator(x):
    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    w0 = tf.get_variable('G_w0', [x.get_shape()[1], 256], initializer=w_init)
    b0 = tf.get_variable('G_b0', [256], initializer=b_init)
    h0 = tf.nn.leaky_relu(tf.matmul(x, w0) + b0)

    w_1 = tf.get_variable('G_w1', [h0.get_shape()[1], 512], initializer=w_init)
    b_1 = tf.get_variable('G_b1', [512], initializer=b_init)
    h_1 = tf.nn.leaky_relu(tf.matmul(h0, w_1) + b_1)

    w_2 = tf.get_variable('G_w2', [h_1.get_shape()[1], 1024], initializer=w_init)
    b_2 = tf.get_variable('G_b2', [1024], initializer=b_init)
    h_2 = tf.nn.leaky_relu(tf.matmul(h_1, w_2) + b_2)

    w_3 = tf.get_variable('G_w3', [h_2.get_shape()[1], 784], initializer=w_init)
    b_3 = tf.get_variable('G_b3', [784], initializer=b_init)
    o = tf.nn.tanh(tf.matmul(h_2, w_3) + b_3)

    # hidden layers w_0 > w_1 > w_2 > w_3

    return o


def discriminator(x, drop_out):
    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    ###  Code: ToDO( Change the architecture as CW2 Guidance required)
    w0 = tf.get_variable('D_w0', [x.get_shape()[1], 784], initializer=w_init)
    b0 = tf.get_variable('D_b0', [784], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

    w_1 = tf.get_variable('D_w1', [h0.get_shape()[1], 1024], initializer=w_init)
    b_1 = tf.get_variable('D_b1', [1024], initializer=b_init)
    h_1 = tf.nn.leaky_relu(tf.matmul(h0, w_1) + b_1)
    dropped_h1 = tf.nn.dropout(h_1, 0.3)

    w_2 = tf.get_variable('D_w2', [dropped_h1.get_shape()[1], 512], initializer=w_init)
    b_2 = tf.get_variable('D_b2', [512], initializer=b_init)
    h_2 = tf.nn.leaky_relu(tf.matmul(dropped_h1, w_2) + b_2)
    dropped_h2 = tf.nn.dropout(h_2, 0.3)

    w_3 = tf.get_variable('D_w3', [dropped_h2.get_shape()[1], 256], initializer=w_init)
    b_3 = tf.get_variable('D_b3', [256], initializer=b_init)
    h_3 = tf.nn.leaky_relu(tf.matmul(dropped_h2, w_3) + b_3)
    dropped_h3 = tf.nn.dropout(h_3, 0.3)

    w_4 = tf.get_variable('D_w4', [dropped_h3.get_shape()[1], 1], initializer=w_init)
    b_4 = tf.get_variable('D_b4', [1], initializer=b_init)
    o_4 = tf.sigmoid(tf.matmul(dropped_h3, w_4) + b_4)
    
    # hidden layers w_0 > w_1 > w_2 > w_3

    return o_4


fixed_z_ = np.random.normal(0, 1, (25, 100))


def show_result(num_epoch, k, show=True, save=True, path='result.png', fixed=False):
    z_ = np.random.normal(0, 1, (25, 100)) 
    ##Code:ToDo complete the rest of part

    if not fixed:
        test_images = sess.run(G_z, {z: z_, drop_out: 0.0})
    else:
        test_images = sess.run(G_z, {z: fixed_z_, drop_out: 0.0})

    size = 5
    fig, ax = plt.subplots(size, size, figsize=(5, 5))
    for i, j in itertools.product(range(size), range(size)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    # k = 0
    while k < size**2:
        i = k // 5
        j = k % 5
        x[i, j]
        ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')
        k = k + 1
        
    label = 'Epoch ' + str(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
      plt.show()
    else:
      plt.close()

def show_train_hist(hist, show = True, save = True, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# training parameters
batch_size = 100
lr = 0.0002
train_epoch = 10

# load MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1 ~ 1

# networks : generator
with tf.variable_scope('G'):
    z = tf.placeholder(tf.float32, shape=(None, 100))
    G_z = generator(z)
# networks : discriminator
with tf.variable_scope('D') as scope:
    drop_out = tf.placeholder(dtype=tf.float32, name='drop_out')
    x = tf.placeholder(tf.float32, shape=(None, 784))
    D_real = discriminator(x, drop_out)
    scope.reuse_variables()
    D_fake = discriminator(G_z, drop_out)
# loss for each network
eps = 1e-2
D_loss = tf.reduce_mean(-tf.log(D_real + eps) - tf.log(1 - D_fake + eps))
G_loss = tf.reduce_mean(-tf.log(D_fake + eps))

# trainable variables for each network
t_vars = tf.trainable_variables()
D_vars = [var for var in t_vars if 'D_' in var.name]
G_vars = [var for var in t_vars if 'G_' in var.name]

# optimizer for each network
D_optim = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=D_vars)
G_optim = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# results save folder
if not os.path.isdir('MNIST_GAN_results'):
    os.mkdir('MNIST_GAN_results')
if not os.path.isdir('MNIST_GAN_results/results'):
    os.mkdir('MNIST_GAN_results/results')
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
# training-loop
np.random.seed(int(time.time()))
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in range(train_set.shape[0] // batch_size):
        # update discriminator
        x_ = train_set[iter * batch_size:(iter + 1) * batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 100))

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, drop_out: 0.3})
        D_losses.append(loss_d_)

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 100))
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, drop_out: 0.3})
        G_losses.append(loss_g_)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))

    ### Code: TODO Code complete show_result function)

    my_path = 'MNIST_GAN_results/results/' + str(epoch + 1) + '.png'
    show_result((epoch + 1), 0, show=False, save=True, path=my_path, fixed=False)

    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)
print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open('MNIST_GAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)
show_train_hist(train_hist, save=True, path='MNIST_GAN_results/MNIST_GAN_train_hist.png')
images = []
sess.close()
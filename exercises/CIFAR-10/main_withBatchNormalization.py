import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta
import cifar10

#cifar10.download()

print(cifar10.load_class_names())

train_img, train_cls, train_labels = cifar10.load_training_data()
test_img, test_cls, test_labels = cifar10.load_test_data()

print("Train Image:", len(train_img), "Test Image: ", len(test_img))

x = tf.placeholder(tf.float32, [None, 32, 32, 3]) # x = 32 y = 32 Channel = 3 (RGB)
y_true = tf.placeholder(tf.float32, [None, 10]) # Classes = 10
pkeep = tf.placeholder(tf.float32)
phase = tf.placeholder(tf.bool)

def pre_process_image(image):#Data Augmentation Part
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, max_delta = 0.05)
    image = tf.image.random_contrast(image, lower = 0.3, upper = 1.0)
    image = tf.image.random_brightness(image, max_delta = 0.2)
    image = tf.image.random_saturation(image, lower = 0.0, upper = 2.0)

    image = tf.minimum(image, 1.0)
    image = tf.maximum(image, 0.0)
    return image

def pre_process(images):
    images = tf.map_fn(lambda image: pre_process_image(image), images)
    return images

with tf.device('/cpu:0'):
    distorted_images = pre_process(images = x)

def batch_normalization(input, phase, scope):
    return tf.cond(phase, 
                        lambda: tf.contrib.layers.batch_norm(input, decay = 0.99, is_training = True, updates_collections = None, center = True, scope = scope),
                        lambda: tf.contrib.layers.batch_norm(input, decay = 0.99, is_training = False, updates_collections = None, center = True, scope = scope, reuse = True))

def conv_layer(input, size_in, size_out, scope, use_pooling = True):
    w = tf.Variable(tf.truncated_normal([3, 3, size_in, size_out], stddev = 0.1))
    b = tf.Variable(tf.constant(0.1, shape = [size_out]))

    conv = tf.nn.conv2d(input, w, strides = [1, 1, 1, 1], padding = 'SAME') + b
    conv_bn = batch_normalization(conv, phase, scope)
    y = tf.nn.relu(conv)

    if use_pooling:
        y = tf.nn.max_pool(y, ksize = [1, 2, 2 ,1], strides = [1, 2, 2, 1], padding = 'SAME')

    return y 

def fc_layer(input, size_in, size_out, scope, relu = True, dropout = True, batch_norm = False):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev = 0.1))
    b = tf.Variable(tf.constant(0.1, shape = [size_out]))
    logits = tf.matmul(input, w) + b
    if batch_norm:
        logits = batch_normalization(logits, phase, scope)
    if relu:
        y = tf.nn.relu(logits)
        if dropout:
            y = tf.nn.dropout(y, pkeep)
        return y
    else:
        return logits

#input = [32, 32, 3]
#conv1 = conv_layer(x, 3, 32, use_pooling = True) #output = [16, 16, 32] #without Data Augmentation
conv1 = conv_layer(distorted_images, 3, 32, scope = 'conv1', use_pooling = True) #output = [16, 16, 32] #with Data Augmentation
conv2 = conv_layer(conv1, 32, 64, scope = 'conv2', use_pooling = True) #output = [8, 8, 64]
conv3 = conv_layer(conv2, 64, 64, scope = 'conv3', use_pooling = True) #output = [4, 4, 64]

flattened = tf.reshape(conv3, [-1, 4 * 4 * 64])

fc1 = fc_layer(flattened, 4 * 4 * 64, 512, scope = 'fc1', relu = True, dropout = True, batch_norm = True)
fc2 = fc_layer(fc1, 512, 256, scope = 'fc2', relu = True, dropout = True, batch_norm = True)
logits = fc_layer(fc2, 256, 10, scope = 'fc_out', relu = False, dropout = False, batch_norm = False)
y = tf.nn.softmax(logits)

y_pred_cls = tf.argmax(y, 1)

xent = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_true)
loss = tf.reduce_mean(xent)

correct_prediction = tf.equal(y_pred_cls, tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(5e-4).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128

def random_batch():
    index = np.random.choice(len(train_img), size = batch_size, replace = False)
    x_batch = train_img[index, :, :, :]
    y_batch = train_labels[index, :]

    return x_batch, y_batch

loss_graph = []

def training_step(iterations):
    start_time = time.time()
    for i in range(iterations):
        x_batch, y_batch = random_batch()
        feed_dict_train = {x: x_batch, y_true: y_batch, pkeep: 0.5, phase: True}
        [_, train_loss] = sess.run([optimizer, loss], feed_dict = feed_dict_train)

        loss_graph.append(train_loss)
        
        if i % 10 == 0:
            train_acc = sess.run(accuracy, feed_dict = feed_dict_train)
            print("Iterations: ", i, "Training accuracy: ", train_acc, "Training loss: ", train_loss)
    
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: ", timedelta(seconds = int(round(time_dif))))

batch_size_test = 256

def test_accuracy():
    num_images = len(test_img)
    cls_pred = np.zeros(shape = num_images, dtype = np.int)
    i = 0
    while i < num_images:
        j = min(i + batch_size_test, num_images)
        feed_dict = {x: test_img[i:j, :], y_true: test_labels[i:j, :], pkeep: 1, phase: False}
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict = feed_dict)
        i = j
    correct = (test_cls == cls_pred)
    print("Testing accuracy: ", correct.mean())

training_step(100000)
test_accuracy()

plt.plot(loss_graph, 'k-')
plt.title("Loss Graph")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
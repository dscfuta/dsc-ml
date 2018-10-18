import tensorflow as tf
import os
from collections import deque
import numpy as np


filenames_train=[i for i in map(lambda x :"all\\train\\divide\\"+x ,
                        os.listdir("all\\train\\divide"))]



filenames_test=[i for i in map(lambda x :"all\\test1\\test1\\"+x ,
                        os.listdir("all\\test1\\test1"))]
print(filenames_test[0])
BATCH=10
EPOCH=1
LAYERS=4

def parse_image(filename, label=None):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [150, 150])
  if label is None:
    return image_resized
  return image_resized, label

def parselabels(name):
    return 1 if name.split(".")[-3][-3:]=="cat" else 0


labels_train = tf.constant([parselabels(name) for name in filenames_train])


filenames_train = tf.constant(filenames_train)
filenames_test = tf.constant(filenames_test)

dataset_train = tf.data.Dataset.\
            from_tensor_slices((filenames_train, labels_train)).map(parse_image).\
                batch(BATCH).repeat(EPOCH).\
                shuffle(buffer_size=100)

dataset_test = tf.data.Dataset.\
            from_tensor_slices(filenames_test).map(parse_image).\
                batch(BATCH)



iterator_train = dataset_train.make_initializable_iterator()
iterator_test=dataset_test.make_initializable_iterator()

next_element_train = iterator_train.get_next()
next_element_test=iterator_test.get_next()



input=tf.placeholder(tf.float32, [None, 150,150,3],name="input")

layers=deque([input])

filters=[32,64,128,128]

for _ in range(LAYERS):
    #The covulutional layer of our network
    conv = tf.layers.conv2d(
            inputs=layers[-1],
            filters=filters[_], #Filters define the number of network parameter to learn
            kernel_size=[3, 3], #kernel_size define the size of a convulution 
            padding="valid",
            activation=tf.nn.relu,
            name="conv"+str(_))
    
    pool = tf.layers.average_pooling2d(inputs=conv,
                                 pool_size=[2, 2],
                                  strides=2,
                                  name="pool"+str(_))
    print(pool.shape)
    layers.append(pool)
    layers.popleft()
        
reshape=tf.reshape(layers[-1],[-1,128*128*128],name="reshapelayer")

dense=tf.layers.dense(reshape,512,name="dense", activation=tf.nn.relu)

output=tf.layers.dense(dense,1,name="finaloutput",activation=tf.nn.sigmoid)

prediction = tf.placeholder(tf.float32, [None,1])


#loss
loss = tf.losses.softmax_cross_entropy(prediction,output)
train = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss)


with tf.Session() as sess:
    tf.initialize_all_variables().run()
    sess.run(iterator_train.initializer)
    while True:
      try:
        next_batch=sess.run(next_element_train)
        loss_,train_=sess.run([loss,train],feed_dict={
            input:next_batch[0],
            prediction:np.reshape(next_batch[1],[-1,1])
        })
        print(loss_)
      except tf.errors.OutOfRangeError:
        break

    sess.run(iterator_test.initializer)

    next_batch=sess.run(next_element_test)
    print(sess.run(output,feed_dict={input:next_batch}))
    

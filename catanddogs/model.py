import tensorflow as tf
import imagetotensor
import numpy as np
import os
from collections import deque


#THE Number of layer of our intented network
LAYERS=4
BATCHES=5

def main():
    inputBatch=tf.placeholder(tf.float32, [None, 50,50,3],name="input")
    print(inputBatch.shape)
    layers=deque([inputBatch])
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
    
        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2,name="pool"+str(_))
        print(conv.shape,pool.shape)
        layers.append(pool)
        layers.popleft()
        
    reshape=tf.reshape(layers[-1],[-1,128],name="reshapelayer")
    dense=tf.layers.dense(reshape,50,name="dense", activation=tf.nn.relu)
    output=tf.layers.dense(dense,1,name="finaloutput",activation=tf.sigmoid)
    print(output.shape)
        
    prediction = tf.placeholder(tf.float32, [None,1])
    print(prediction.shape)

    #loss
    loss = tf.reduce_mean(tf.abs(output-prediction))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(loss)

    saver=tf.train.Saver()

    print("fetching image  ")
    INPUTTRAIN=imagetotensor.getTrainbatch()
    catimage,dogimage,catprediction,dogprediction=INPUTTRAIN
    
    lengthofdata=len(catimage) #the length of the data
    numberofbatch=lengthofdata//BATCHES #the number of batches


    catprediction=np.reshape(catprediction,[-1,1])
    dogprediction=np.reshape(dogprediction,[-1,1])
    
    print("Sessoion starting")
    
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        cat_or_dog=True #true to use the cat image false to use the dog image
        for _ in range(BATCHES):
            print("training on "+str(_)+" batch ")
            if cat_or_dog:
                train_train=sess.run(train_step,feed_dict={inputBatch:
                            catimage[_*numberofbatch:_*numberofbatch+numberofbatch],
                            prediction: catprediction[_*numberofbatch:_*numberofbatch+numberofbatch]})
                cat_or_dog=not cat_or_dog
            else:
                train_train=sess.run(train_step,feed_dict={inputBatch:
                            dogimage[_*numberofbatch:_*numberofbatch+numberofbatch],
                            prediction: dogprediction[_*numberofbatch:_*numberofbatch+numberofbatch]})
                cat_or_dog=not cat_or_dog
            saver.save(sess,os.getcwd()+"/modelsave/modelsave")

        #Testing the model
        batchToTest=1
        correct_prediction = tf.equal(output,prediction)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={inputBatch:
                            catimage[batchToTest*numberofbatch:batchToTest*numberofbatch+numberofbatch],
                            prediction: catprediction[batchToTest*numberofbatch:batchToTest*numberofbatch+numberofbatch]}))
                              #prints the accuracy to the console
    
       



if __name__=="__main__":
    main()
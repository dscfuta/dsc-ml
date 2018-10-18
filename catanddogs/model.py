import tensorflow as tf
import pandas as pd
import numpy as np

dataset=pd.read_csv("C:\\Python35\\bootcamp1\\USA_Housing.csv")

def ConvertToInt(column):
    Dict={}
    j=1
    for i in column:
        if i not in Dict:
            Dict[i]=j
            j+=1    
    result=[]
    for i in column:
        result.append(Dict[i])
    return result
X=[];Y=[]

col_y=dataset["Price"]

for _ in range(len(dataset)):
    Y.append(col_y[_])

dataset["Address"]=ConvertToInt(dataset["Address"])

dataset=dataset.drop(columns=["Price"])
for _ in range(len(dataset)):
    X.append(dataset.loc[_,:])

X=np.asarray(X)
print(X.shape)
Y=np.reshape(np.asarray(Y),[-1,1])
print(Y.shape)

input=tf.placeholder(tf.float32,[None,6])
pred=tf.placeholder(tf.float32,[None,1])
layer1=tf.nn.relu(tf.layers.dense(input,64))
layer2=tf.nn.relu(tf.layers.dense(layer1,64))
output=tf.layers.dense(layer2,1)

loss=tf.losses.mean_squared_error(pred,output)

train=tf.train.RMSPropOptimizer(0.01).minimize(loss,
                                                global_step=tf.train.get_global_step())

from random import randint


saver=tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    batch=10
    epoch=3
    for _ in range(epoch):
        total_loss=0
        print("training on batch "+str(_+1))
        for data in range(len(dataset)):
            niner=[randint(0,5000-1) for i in range(9)]
            X_to_feed=[X[data]]+[X[niner[i]] for i in range(9)]
            Y_to_feed=[Y[data]]+[Y[niner[i]] for i in range(9)]

            loss_,train_=sess.run([loss,train],feed_dict={
                input:X_to_feed,
                pred:Y_to_feed
            })
            total_loss+=loss_
        print(total_loss)
    Y_pred=sess.run(output,feed_dict={input:X})

    for i in range(20):
        print(Y_pred[i],Y[i])
    error=0
    for i in range(len(Y_pred)):
        error+=abs(Y[i][0]-Y_pred[i][0])
    print(error)
    
    import os
    saver.save(sess,os.getcwd())
 
import os
import matplotlib.image as img
import numpy as np
import cv2


def ConvertFolderToTensor(path):
    cat=[]
    dogs=[]
    for image in os.listdir(path):
        if image[:3]=="cat":
            cat.append(img.imread(os.path.join(path,image)))
        else:
            dogs.append(img.imread(os.path.join(path,image)))
    return np.asarray(cat),np.asarray(dogs),np.ones(len(cat)),np.zeros(len(dogs))

def getTrainbatch():
    return ConvertFolderToTensor(os.path.join(os.getcwd()+"/all","train","train"))

def getTestBatch():
    return ConvertFolderToTensor("")


def resizeAllInputImages(heigth,width,path):
    for FILE in os.listdir(path):
        print(FILE)
        img = cv2.imread(path+"/"+FILE)
        img=cv2.resize(img,(heigth,width))
        cv2.imwrite(path+"/"+FILE,img)

if __name__=="__main__":
    resizeAllInputImages(50,50,os.path.join(os.getcwd()+"/all","train","train"))
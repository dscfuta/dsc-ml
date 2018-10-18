import tensorflow as tf
import os
import pandas as pd


def readDataSet():
    dataset=pd.read_csv("Flight Delays Data.csv")
    CarrierName=["HA","WN","DL","AA","UA","US","OO","EV","F9","MQ","FL","B6","9E",'AS','VX','YV']
    carriers=dataset["Carrier"]
    finalcarries=[]
    for c in carriers:
        finalcarries.append(CarrierName.index(c)) 
    dataset["Carrier"]=finalcarries
    

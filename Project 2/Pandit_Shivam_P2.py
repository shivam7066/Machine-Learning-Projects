# -*- coding: utf-8 -*-
"""
#Author: Shivam Pandit
#Date: 9/27/2019
#Project 2
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import operator

def load_splitdata(dataset):
    fin = open(dataset, "r")
    rows = len(fin.readlines())
    #print(rows)
    data = np.zeros([rows-1,7])
    #print(data)
    fin.close()
    
    fin = open(dataset,"r")
    line = fin.readline()
    for i in range(rows-1):
        line = fin.readline()
        t = line.strip("\n")   #Removing trailing newline character
        t = t.split("\t")   #Splitting string into list using delimiter \t
        
        for j in range(7):
            
            if j==0:            
                data[i][j] = 1
            elif j == 1:           
                data[i][j] = int(t[0])
            elif j == 2:
                data[i][j] = int(t[1])
            elif j==3:
                data[i][j] = int(t[0]) * int(t[1])
            elif j==4:
                data[i][j] = pow(int(t[0]),2)
            elif j==5:
                data[i][j] = pow(int(t[1]), 2)
            elif j==6: 
                data[i][j] = float(t[2])  
        
    #print(data)
    #assert data.shape == (300,7)
    
    #Randomly Shufle data
    np.random.shuffle(data)
    
    """
    ##Checking one row to see values in data nd array
    for i in range(7):
        print(data[0][i])
    """
    print(data.shape[0])
    
    #Get GPA scores label to keep track and remove it from working dataset
    labels = data[:,6]
    data = data[:,:6]
    
    #print(data[6][5])
    #print(labels)
    
    #Feature Scaling to avoid issues calculating costs
    #Using standarization by subtracting mean and dividing by standard deviation
    data_std = (data - np.mean(data))/np.std(data)
    #data_std = np.round(data_std,3)  #rounding decimal values to 3 places

    
    
    #assert data_std.shape == (300,7)
    
    #SPLITTING INTO TRAIN AND TEST
    split = 0.7
    train_len= int(split*rows)
    train= data_std[:train_len]
    test = data_std[train_len:]
    train_label = labels[:train_len]
    test_label = labels[train_len:]
    
    
    
    #print(train_label)
   
    return train,test,train_label,test_label

def get_costs(m, train, train_label, wt):
    y = np.dot(wt.T , train.T)    
    #print("Hyp Pred. Y \n:",y)    
    cost = 1 / (2 * train.shape[0]) * np.sum(np.power((y - train_label), 2))
    
    return np.squeeze(cost),y

def initial_weights(train, test, train_label, test_label):    
    cost = 0
    sum = 0
    np.random.seed(7)  # Using seed to generate same random nos.
    #Initialize weights randomly using random generator
    wt = np.round(np.random.rand(train.shape[1],1),2)

    #print(wt)
    
    wt_T = wt.T
    #print("Wt T:", wt_T)
    #Wt T: [[0.89 0.33 0.82 0.04 0.11 0.6 ]]    
    """
    y = np.dot(wt_T , train.T)
    print("Hyp Pred.:",y)
    print("True Label:" , train_label[1])
    print("Train Transpose shape:",train.T.shape)
    print("Train1\n",train.T)
    print("TRAIN\n:", train)
    """
    cost,y = get_costs(train.shape[0], train, train_label, wt)    
    
    #print("Initial Cost:", cost)    
    """""""""
    print("TrainLabel:\n",train_label)
    q=y-train_label
    print("Y- Train_Label:\n",q)
    print(q*q)
    sq = np.sum(q*q)
    cq = 1 / (2 * train.shape[0]) * sq
    print("sq:\n",sq)
    print("cq:\n",cq)
    print("Initial Weights:\n",wt)
    """""""""
    
    return cost, wt
    #print(train.shape)

def best_weights(train,test,train_label,test_label,wt,cost):
    
    m =  train.shape[0]
    iterations = 1                                                                                                                                                                                                                                                                                 
    learning_rate = 0.01
    costs = []
    temp = np.zeros([train.shape[1]])
    
    for i in range(iterations):    
             
        y = np.dot(wt.T , train.T)
        print(y)
        loss = y - train_label
        print(loss)
        for j in range(train.shape[1]):         
            temp[j] = wt[j] - learning_rate *(1/m)* np.sum(loss * train.T[j])
            
        new_cost,y= get_costs(m,train,train_label,temp)  
        wt = temp
        costs.append(round(new_cost,9))
        
        #Printing after 50 iterations
        #if(i%50==0):        
            #print("new cost: Iteration{}: {}" .format(i,new_cost))
     
    #print(wt) 
    #print("costs", costs)
    
    """"
      
    print("temp\n",temp)
    print("loss",loss)
    print("train t0",train.T[0])
    print("Test:", loss * train.T[0])    
    print("train:\n", train)
    print("train_T:\n", train.T)
    print("train_T:\n", train[1].T)
        
    print("wt\n",wt)

    print("train label:", train_label)       
            """

    return wt, costs , iterations

def plot_cv(iter, j):
    iter = list(range(0,iter))
    plt.plot(iter, j)
    plt.xlabel("No. of iterations")
    plt.ylabel("Value of J")
    plt.title("Plot of J  vs. Number of Iterations")
    plt.savefig("Pandit_Shivam_Regression_MyPlot.png", bbox_inches="tight")
    
def test_J(test, test_label, wt):
    cost_test,y = get_costs(test.shape[0], test, test_label, wt)
    #print("TestCost: ", cost_test)
    return cost_test
    

def predict(hours,drink,w):
    #w = [-0.02533487, -0.58025172, -0.09189377 , 0.13368376,  0.62353464,  0.30679626]
    x1= int(hours)
    x2= int(drink)
    #Standardizing User input - Feature Scaling
    data = np.zeros([1,7])
    #print(data.shape)

    data1 = [1, x1 , x2, x1*x2 , x1*x1 , x2*x2]  # 1*7
    #print("data1:", data1)
    data = (data1 - np.mean(data1))/np.std(data1)
    y = np.dot(w , data.T)
    
    #print("Input Data:",data)
    #print("mean",np.mean(data))
    #print("Sd",np.std(data))

    #print("wt:\n",w)
    #print("Scaled Data:", data)

    #Predicting after standardizing user input                                                                                                                                                                                                          
    y1 = w[0]*data[0] + w[1]*data[1] + w[2]*data[2] + w[3]*data[3] + w[4]*data[4] + w[5]*data[5]
    #print("Prediction:",round(y,2))  
    #print("Prediction1:",round(y1,2)) 
    #y = round(y,2)
    
    return y

    
def main():

    #file = "GPAData.txt"
    file = "test.txt"
    train,test,train_label,test_label = load_splitdata(file)
    cost, wt= initial_weights(train,test,train_label,test_label)    
    bwt, costs, iter = best_weights(train,test,train_label,test_label,wt,cost)    
    #plot_cv(iter,costs)
    test_cost=test_J(test,test_label,bwt)
    
    y = round(predict(436,166, bwt),2) #For testing predict function
    print(y)    
    
    '''
    while(True):
        hour = input("Enter values for minutes spent studying per week: ")
        hour= int(hour)
        drink = input("Enter ounces of beer consumed per week: ")
        drink= int(drink)
        if hour == 0 and drink == 0:
            break
        prediction = predict(hour,drink,bwt)
        print("Predicted GPA:",round(prediction,2))  
        
        
        '''
main()
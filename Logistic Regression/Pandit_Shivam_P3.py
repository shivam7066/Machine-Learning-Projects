"""
Author: Shivam Pandit
Date: 10/10/2019
Project 3: Logistic Regression for TigerFish
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import operator

#Reading File and splitting into train and test
def load_splitdata(dataset):
    fin = open(dataset, "r")
    rows = len(fin.readlines())
    #print(rows)
    data = np.zeros([rows-1,4])
    #print(data)
    fin.close()
    
    fin = open(dataset,"r")
    line = fin.readline()
    for i in range(rows-1):
        line = fin.readline()
        t = line.strip("\n")   #Removing trailing newline character
        t = t.split("\t")   #Splitting string into list using delimiter \t
        
        for j in range(4):
            
            if j==0:            
                data[i][j] = 1
            elif j == 1:           
                data[i][j] = float(t[0]) 
            elif j == 2:
                data[i][j] = float(t[1])
            elif j==3:
                data[i][j] = int(t[2])
        

    #Check dimensions
    assert data.shape == (300,4)
    
    #Randomly Shufle data
    np.random.shuffle(data)

    #Get predcitions label to keep track and remove it from working dataset
    labels = data[:,3]
    data = data[:,:3]
    
    #Feature Scaling to avoid issues calculating costs
    #Using standarization by subtracting mean and dividing by standard deviation
    data_std = (data - np.mean(data))/np.std(data)
    #data_std = np.round(data_std,3)  #rounding decimal values to 3 places
   
    assert data_std.shape == (300,3)
    
    #SPLITTING INTO TRAIN AND TEST
    split = 0.7
    train_len= int(split*rows)
    train= data_std[:train_len]
    test = data_std[train_len:]
    train_label = labels[:train_len]
    test_label = labels[train_len:]
   
    return train,test,train_label,test_label

'''
""""""""""""""""""""""""""""""""""""""""
PLOTTING DATASET
"""""""""""""""""""""""""""""""""""""""
#Plot the data points with color coding for TigerFish1 and TigerFish0
for k in range(rows-1):
    if data[k,2] == 0:
        x=plt.scatter(data[k,0], data[k,1], marker='v', color = "blue", label = "TigerFish0")
   
    else:
        y=plt.scatter(data[k,0], data[k,1], marker='o', color = "green", label = "TigerFish1")

#Adding marker legend table inside plot box
plt.legend((x, y),
           ( "TigerFish0", "TigerFish1"),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=9)   

#Labelling Plot axes and title 
plt.xlabel("Body Length")
plt.ylabel("Dorsal Fin Length")
plt.title("Scatter Plot for Tiger Fish")

#Saving plot to png file format
#plt.savefig("Pandit_Shivam_MyPlot.png", bbox_inches="tight")
plt.show()
'''

def get_costs(m, train, train_label, wt):

    cost = 1
    m = train.shape[0]
    y = np.dot(wt.T , train.T)                           
    #print("Dot Product \n:",y)    
    y = 1 / (1 + np.exp(-y)) 
    cost = -(1/m) * np.sum(train_label*np.log(y) + (1-train_label)*np.log(1-y))

    return np.squeeze(cost),y

def initial_weights(train, test, train_label, test_label):    
    cost = 0
    sum = 0
    np.random.seed(2)  # Using seed to generate same random nos.
    #Initialize weights randomly using random generator
    wt = np.round(np.random.rand(train.shape[1],1),2)
    #print(wt)    
    wt_T = wt.T
    #print("Wt T:", wt_T)
    #Wt T: [[0.55 0.71 0.29]]   
    cost,y = get_costs(train.shape[0], train, train_label, wt)     
    #print("Initial Cost:", cost)   
    return cost, wt


def test_J(test, test_label, wt):
    cost_test,y = get_costs(test.shape[0], test, test_label, wt)
    #print("TestCost: ", cost_test)
    return cost_test
    

def best_weights(train,test,train_label,test_label,wt,cost):
    
    m =  train.shape[0]
    iterations = 1000                                                                                                                                                                                                                                                                                
    learning_rate = 0.01
    costs = []
    new_cost = 1
    #print(train.T)
    temp = np.zeros([train.shape[1]])
    
    for i in range(iterations):    
             
        y = np.dot(wt.T , train.T)
        y = 1 / (1 + np.exp(-y))       
        loss = y - train_label
        
        for j in range(train.shape[1]):         
            temp[j] = wt[j] - learning_rate * np.sum(loss * train.T[j])
        
        #print(temp)
        new_cost,y= get_costs(m,train,train_label,temp)  
        wt = temp
        costs.append(round(new_cost,3))
        '''
        #Printing after 10 iterations
        if(i%100==0):        
            print("new cost: Iteration{}: {}" .format(i,new_cost))
        '''
    #print("Final Cost on Training Set:", costs[-1])
    #prediction_train = accuracy(wt, train, train_label)
    #prediction_test = accuracy(wt, test, test_label)
    
    #print Accuracy
    #print("Train Accuracy: " , prediction_train)
    #print("Test Accuracy: ",prediction_test)
    #print("Best Weight:", wt)     
    
    return wt, costs , iterations

def accuracy(w , data, data_label):
    y = np.dot(w.T , data.T)                      
    y = 1 / (1 + np.exp(-y))     
    acc = 100 - np.mean(np.abs(y - data_label)) * 100    
    return acc    
    
def plot_cv(iter, j):
    iter = list(range(0,iter))
    plt.plot(iter, j)
    plt.xlabel("No. of iterations")
    plt.ylabel("Value of J")
    plt.title("Plot of J  vs. Number of Iterations")
    plt.savefig("Pandit_Shivam_Regression_MyPlot.png", bbox_inches="tight")

def predict(body, fin, w):
    #w = [  4.35008813 , -6.66381348 , -22.87915831]
    x1= float(body)
    x2= float(fin)
    #Standardizing User input - Feature Scaling
    data = np.zeros([1,3])
    data1 = [1, x1 , x2]  # 1*3
    data = (data1 - np.mean(data1))/np.std(data1)
    y = np.dot(w, data.T)
    y = 1 / (1 + np.exp(-y))                                                                                                                                                                                                
   
    if y>0.5:
        return "TigerFish1"    
    else:
        return "TigerFish0"

def confusion_matrix(wt, data, data_label):
    y = np.dot(wt , data.T)                      
    y = 1 / (1 + np.exp(-y))
    tp = 0;  tn = 0;  fp = 0;  fn = 0
    predict = []
    for i in range(y.shape[0]):
        if y[i] >= 0.5:
            predict.append(1)
        else:
            predict.append(0)

    for k in range(len(predict)):
        if predict[k] == 1 and data_label[k] == 1:
            tp += 1
        elif predict[k] == 0 and data_label[k] == 0:
            tn += 1
        elif predict[k] == 0 and data_label[k] == 1:
            fn += 1
        elif predict[k] == 1 and data_label[k] == 0:
            fp += 1
    print("True Positives: {}, True Negatives: {}, False Positives: {}, False Negatives: {}".format(tp, tn, fp, fn))
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (1 / ((1 / precision) + (1 / recall)))
    print("Accuracy: ", round(accuracy,2))
    print("Precision: ", round(precision,2))
    print("Recall: ", round(recall,2))
    print("f1_score: ", round(f1_score,2))

def main():
    file = "FF71.txt"
    train,test,train_label,test_label = load_splitdata(file)
    cost, wt= initial_weights(train,test,train_label,test_label)    
    bwt, costs, iter = best_weights(train,test,train_label,test_label,wt,cost) 
    #plot_cv(iter,costs)
    test_cost=test_J(test,test_label,bwt)
    #confusion_matrix(bwt, test, test_label)
    #y = predict(61,7, bwt) #For testing predict function
    #print(y)        
    
    while(True):
        b_length = input("Enter Body Length(in centimeters): ")
        b_length = float(b_length)
        df_length = input("Enter Dorsal Fin Length(in centimeters)3: ")
        df_length = float(df_length)
        if df_length == 0 and b_length == 0:
            break
        prediction = predict(b_length, df_length, bwt)
        print(prediction)   
        
main()
    
# -*- coding: utf-8 -*-
"""
#Author: Shivam Pandit
#Date: 9/15/2019
#Project 1
#Description: Given a data set representing given the body length and dorsal fin length of a fish,
#we have to create a k-Nearest Neighbor program that will predict if it is TigerFish1 or TigerFish0.
"""
#Import libraries for array objects and plotting
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import operator

random.seed(1)

#Reading FF71.txt file and creating nd array with dimensions rows * 5
fin = open("FF71.txt", "r")
rows=len(fin.readlines())
#print(rows)
data=np.zeros([rows-1,3], dtype=object)
fin.close()

#Read data from the file and put it into the train and test after randomizing
def splitDataset(file, split, train=[] , test=[]):
    fin = open(file , "r")
    line= fin.readline()        #Removing first line that mentions records in dataset	
     
    for k in range(rows-1):            
        line = fin.readline()            
        t = line.strip("\n")     #Removing trailing newline character
        t = t.split("\t")        #Splitting string into list using delimiter \t
        
        for j in range(len(t)):
            
            if j!=2:
                data[k][j] = float(t[j])
                
            else:
                data[k][j] = int(t[j])
      
    #print("Data Before Shuffle:\n", data)
    
    #Randomizing Data
    np.random.shuffle(data)
    
    #print("Data After Shuffle:\n", data)
    
    #SPLITTING INTO TRAIN AND TEST
    train_len= int(split*rows)
    train= data[:train_len]
    test = data[train_len:]
  
    #print("TRAIN:\n", train)
    #print("TEST:\n", test)
    
    return train, test

#Spltiing data into 80:20 train:test      
train=[]
test=[]
train, test = splitDataset("FF71.txt", 0.80, train, test)
#print(len(test))

""""""""""""""""""""""""""""""""""""""""
PLOTTING DATASET
"""""""""""""""""""""""""""""""""""""""
#Plot the data points with color coding for Setosa, Virginica & Versicolor
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
#plt.show()

"""""""""""""""""""""""""""
Creating K-FOLDS
"""""""""""""""""""""""""""
from random import randrange
random.seed(1)
# Split a dataset into k folds
def k_foldcv(data, folds=5):
	data_fold = list()
	data = data.tolist()
	fold_size = int(len(data) / folds)
    
	for i in range(folds):
		fold_data = list()
		while len(fold_data) < fold_size:
			index = randrange(len(data))
			fold_data.append(data.pop(index))
		data_fold.append(fold_data)
	return data_fold

# test f-fold cross validation split
#folds = k_foldcv(train, 5)

#print(len(folds))
#print(folds[0])
#print(folds[0][47])
"""""""""""""""""""""
Calculate Euclidean Distance
"""""""""""""""""""""""
def euclidean_distance(p1 , p2 , q1 , q2):
    distance = math.pow(p1-q1,2)+ math.pow(p2-q2,2)
    distance = np.sqrt(distance)
    return distance

"""""""""""""""""""""
Calculating k for KNN
"""""""""""""""""""""""
#print(len(test))

#Creating 5 train and vaildation sets for k-fold CV

"""
val1 = folds[0]
train1 = folds[1] + folds[2] + folds[3] + folds[4]
val2 = folds[1]
train2 = folds[0] + folds[2] + folds[3] + folds[4]
val3 = folds[2]
train3 = folds[0] + folds[1] + folds[3] + folds[4]
val4 = folds[3]
train4 = folds[0] + folds[1] + folds[2] + folds[4]
val5 = folds[4]
train5 = folds[0] + folds[1] + folds[2] + folds[3]
"""

def get_distances(train, val, k):
    
    n_distance = []
    distance =[]

    for i in range(len(val)):
        distances = []
               
        for j in range(len(train)):
            p1 = val[i][0]
            p2 = val[i][1]
            q1 = train[j][0]
            q2 = train[j][1]
            dist = round(euclidean_distance(p1, p2, q1, q2),3)
            distances.append((train[j],dist))
           
        n_distance.append(distances)
        
    n_distance.sort(key=operator.itemgetter(1))

    neighbor = []
    for z in range(k):
        neighbor.append(n_distance[z][0][0])
    return neighbor, n_distance

#n,d = get_distances(train2, val2, k=3)
#print(len(d))
#print(d)


def misclassify(data):
    point0 = data[1]
    point0.sort(key=operator.itemgetter(1))
    sorted_p = []
    k_errors = []
    fold = {}
    #print(sorted_d)
    for z in range(len(point0)):
        sorted_p.append(point0[z][0])

    
    for k in range(1,21,2):
        sum_labels=0
        avg = sum_labels/k
        s = sorted_p[:k]
        
        for i in range(len(s)):
            sum_labels+= s[i][2] 
                        
        avg = sum_labels/k
        if avg>0.5:
            p = 1   #Returns majority label
        else:
            p = 0
        
        for i in range(len(s)):
            if s[i][2] != val2[0][2]:
                if "k"+str(k) in fold:
                    fold["k"+ str(k)] += 1
                else:
                    fold["k"+ str(k)] = 1
    
    k_errors.append(fold)
    print(k_errors)    

    return k_errors, sorted_p

#c, s = misclassify(d)
#print(s[0][1])
#print(s)
#print(c[1[0]])


def getResponse(neighbors):
    sum_labels = 0
    # get prediction for k
    for i in range(len(neighbors)):
        sum_labels+= n[i][0]    
    avg = sum_labels/k
    if avg>0.5:
        return 1   #Returns majority label
    else:
        return 0

#test = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
#response= getResponse(n)
#print(neighbors[2][-1])
#print(response)


def plot_cv(kval, error_perc):
    plt.plot(kval, error_perc)
    plt.xlabel("Value of K of KNN")
    plt.ylabel("Cross-Validation Accuracy")
    plt.title("Cross Validation Accuracy Plot")
    plt.savefig("Pandit_Shivam_KnnAccuracy_MyPlot.png", bbox_inches="tight")

#errors_sum = [35, 32, 28, 31, 28, 27, 30, 30, 29, 29, 30]
#k_errors_perc = [85.42, 86.62, 88.33, 87.08, 88.33, 88.75, 87.5, 87.5, 87.91, 87.92, 87.5]
#k_val = list(range(1, 22, 2))
#plot_cv(k_val, k_errors_perc)
    

# Prediction function For project 1 
def predict(p1, p2, data, k):
    distances = []
    closest =[]
    for i in range(len(data)):
        q1 = data[i][0]
        q2 = data[i][1]
        d = euclidean_distance(p1, p2, q1, q2)
        distances.append((data[i][2], d))        
    distances.sort(key=operator.itemgetter(1))
    #print(distances)
    closest = distances[:k]  # Getting K points from distances list
    sum_labels = 0
    # get prediction for k
    for i in range(len(closest)):
        sum_labels+= closest[i][0]    
    avg = sum_labels/k
    
    if avg>0.5:
        return "TigerFish1"    
    else:
        return "TigerFish0"

#p=predict(12,52.2,train,11)
#print(p)

"""""""""""""""""""""""""""""""""""""""""""""""""""""
#Program to ask user inputs and give prediction
# Test with user data for k=11 we got after 5Fold CV
"""""""""""""""""""""""""""""""""""""""""""""""""""""
def main():
    file = input("Enter file name: ")
    fin = open(file, "r")
    rows=len(fin.readlines())
    #print(rows)
    data=np.zeros([rows-1,3], dtype=object)
    
    fin.close()
    
    fin = open(file , "r")
    line= fin.readline()        #Removing first line that mentions records in dataset	
     
    for k in range(rows-1):            
        line = fin.readline()            
        t = line.strip("\n")     #Removing trailing newline character
        t = t.split("\t")        #Splitting string into list using delimiter \t
        
        for j in range(len(t)):
            
            if j!=2:
                data[k][j] = float(t[j])            
            else:
                data[k][j] = int(t[j])
    
    while(True):
        b_length = input("Enter Body Length: ")
        b_length = float(b_length)
        df_length = input("Enter Dorsal Fin Length: ")
        df_length = float(df_length)
        if df_length == 0 and b_length == 0:
            break
        prediction = predict(b_length, df_length, data, k=11)
        print(prediction)

#Execute main function        
main()
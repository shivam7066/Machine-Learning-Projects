# -*- coding: utf-8 -*-
"""
Author: Shivam Pandit
Date: 9/4/2019
Project 0 - Python program that does a single plot of sepal length versus petal length for all three varieties of iris flowers
"""
#Import libraries for array objects and plotting
import numpy as np
import matplotlib.pyplot as plt

#Reading irisdata.txt file and creating nd array with dimensions rows * 5
fin = open("IrisData.txt", "r")
rows=len(fin.readlines())
#print(rows)
data=np.zeros([rows,5], dtype=object)
fin.close()

#Read data from the file and put it into the array 
fin = open("IrisData.txt", "r")

for k in range(rows):
    line = fin.readline()
    t = line.strip("\n")     #Removing trailing newline character
    t = t.split("\t")        #Splitting string into list using delimiter \t

    for j in range(len(t)):
            if j!=4:
                data[k,j] = float(t[j])

            else:
                data[k,j]=t[j]
           
#Plot the data points with color coding for Setosa, Virginica & Versicolor
for k in range(rows):
    if data[k,4] == "setosa":
        x=plt.scatter(data[k,0], data[k,2], marker='v', color = "red", label = "Setosa")
        
    elif data[k,4] == "versicolor":
        y=plt.scatter(data[k,0], data[k,2], marker='^', color = "blue", label = "Versicolor")
    
    else:
        z=plt.scatter(data[k,0], data[k,2], marker='o', color = "green", label = "Virginica")

#Adding marker legend table inside plot box
plt.legend((x, y, z),
           ('Setosa', 'Versicolor', 'Virginica'),
           scatterpoints=1,
           loc='top left',
           ncol=1,
           fontsize=9)   

#Labelling Plot axes and title 
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Scatter Plot for Iris Flower Dataset")

#Saving plot to png file format
plt.savefig("Pandit_Shivam_MyPlot.png", bbox_inches="tight")
plt.show()

#Close the file
fin.close()

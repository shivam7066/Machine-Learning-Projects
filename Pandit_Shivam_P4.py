"""
Author: Shivam Pandit
Date: 10/24/2019
Project 4: Naive Bayes for spam/ham 
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import operator

def vocabulary(train,stop):
    spam = 0
    ham = 0
    counted = dict()
    stopw = set([])

    #creating stop words set
    with open(stop, 'r') as f:
        for line in f:
            if line!="\n":
                word = line.strip()
                stopw.add(word)
    f.close()
  
    fin = open(train, "r")
    textline = fin.readline() 
    
    while textline != "":
        is_spam = int(textline[:1]) 
        
        if is_spam == 1:
            spam = spam + 1 
        
        else:
            ham = ham + 1
            
        textline = cleantext(textline[1:])                 
       
        words = remove_stop_words(textline, stopw)
        print(words)
  
        counted = countwords(words, is_spam, counted) 
        textline = fin.readline()
    print(counted)
    vocab = make_percent_list(1, counted, spam, ham) 
    print(vocab)
    print("Ham:",ham)
    print("Spam", spam)
    fin.close()
    
    return vocab, spam, ham


def cleantext(text):
    text = text.lower() 
    text = text.strip() 
    
    for letters in text:    
        if letters in """[]!.,"-!—@;':#$%^&*()+/?""": 
            text = text.replace(letters, " ")
    return text

def remove_stop_words(words, stop):
    words = words.split()
    words = set(words)
    words = words.difference(stop)
    
    return words

def countwords(words, is_spam, counted):
    for each_word in words:       
        if each_word in counted: 
            
            if is_spam == 1:
                counted[each_word][1]=counted[each_word][1] + 1 
 
            else:
                counted[each_word][0]=counted[each_word][0] + 1

        else:
            if is_spam == 1: 
                counted[each_word] = [0,1] #[ham, spam]
            else:
                counted[each_word] = [1,0]
    return counted

def make_percent_list(k, theCount, spams, hams):
    for each_key in theCount:
            
            theCount[each_key][0] = (theCount[each_key][0] + k)/(2*k+hams) 
            
            theCount[each_key][1] = (theCount[each_key][1] + k)/(2*k+spams)
    return theCount

def testvocab(test,stop,train_vocab, spam, ham):
    testspam = 0
    testham = 0
    counted = dict()
    stopw = set([])

    #creating stop words set
    with open(stop, 'r') as f:
        for line in f:
            if line!="\n":
                word = line.strip()
                stopw.add(word)
    f.close()
    tp=0 ; tn=0; fp=0; fn=0;
    pspam = spam/(spam + ham)
    pham = ham / (spam + ham)
    print("P_Spam:", pspam)
    print("P_Ham:", pham)
    fin = open(test, "r")
    textline = fin.readline() 
    
    while textline != "":
        is_spam = int(textline[:1]) 
        
        if is_spam == 1:
            testspam = testspam + 1 
        
        else:
            testham = testham + 1
            
        textline = cleantext(textline[1:])   
        words = remove_stop_words(textline, stopw)
        print(words)
        p1 , p2 = checkvocab(words , train_vocab)
        print("P1:", p1)
        print("P2:", p2)
        predict = round(compute_probability(p1, p2, pspam, pham ),3)
        print("Predict:", predict)
        
        if predict >= 0.5 and is_spam == 1:
            tp +=1
        elif predict >= 0.5 and is_spam == 0:
            fp+=1
        elif predict <= 0.5 and is_spam == 1:
            fn+=1
        elif predict <= 0.5 and is_spam == 0:
            tn+=1
        
        textline = fin.readline()
    ptest_spam = testspam/(testspam + testham)
    ptest_ham = testham / (testspam + testham)
    print("Total Spam emails in Test set: ", testspam)
    print("Total Ham emails in Test set: ", testham)    
    print("True Positives: {}, True Negatives: {}, False Positives: {}, False Negatives: {}".format(tp, tn, fp, fn))
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (1 / ((1 / precision) + (1 / recall)))
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("f1_score: ", f1_score)
    fin.close()

def compute_probability(p1 , p2, ps, ph):
    print("p1: {} p2: {} ps: {} ph:{}"  .format(p1, p2, ps, ph))
    pf = (p1 * ps) / (( p1 * ps ) + ( p2 * ph ))
    
    return pf

def checkvocab(words, vocab):
    p1 = 1
    p2 = 1
    
    for key, values in vocab.items():
        if key in words:
            p1 *= values[1]
            p2 *= values[0]
        
        else: 
            p1 *= (1 - values[1])
            p2 *= (1 - values[0])
    return p1,p2
    
    
def main():
    train = "GEASTrain.txt"
    stop = "StopWords.txt"
    test = "GEASTest.txt"
    
    #train = input("Enter name of training set file:")
    #stop = input("Enter name of stop words file:)
    #test = input("Enter name of test words file:)
    
    #Create Train Vocabulary    
    train_vocab, spam, ham = vocabulary(train,stop)
    
    testvocab(test,stop, train_vocab, spam, ham)
        
     
main()
    
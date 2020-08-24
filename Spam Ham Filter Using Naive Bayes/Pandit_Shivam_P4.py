"""
Author: Shivam Pandit
Date: 10/24/2019
Project 4: Naive Bayes for spam/ham filter
"""
import math

def vocabulary(train,stop):
    spam = 0
    ham = 0
    counted = dict()
  
    fin = open(train, "r")
    textline = fin.readline() 
    
    while textline != "":
        is_spam = int(textline[:1]) 
        
        if is_spam == 1:
            spam = spam + 1 
        
        else:
            ham = ham + 1
            
        textline = cleantext(textline[1:])      #Removes special characters and symbols
        words = remove_stop_words(textline, stop)  #Removes stopwords
        #print(words)  
        counted = countwords(words, is_spam, counted) 
        textline = fin.readline()
    #print(counted)
    vocab = make_percent_list(0.4, counted, spam, ham) # k=0.4 gave best performance
    #print(vocab)
    #print("Ham:",ham)
    #print("Spam", spam)
    fin.close()
    print("Vocabulary Created!!")
    return vocab, spam, ham


def cleantext(text):
    text = text.lower() 
    text = text.strip() 
    
    for letters in text:    
        if letters in """[]!.,"-!—@;':#$%^&*()+/?""": 
            text = text.replace(letters, " ")
    return text

def remove_stop_words(words, stop):    
    stopw = set([])
    #creating stop words set
    with open(stop, 'r') as f:
        for line in f:
            if line!="\n":
                word = line.strip()
                stopw.add(word)
    f.close()
    
    words = words.split()
    words = set(words)
    words = words.difference(stopw)
    
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

    tp=0 ; tn=0; fp=0; fn=0;
    pspam = spam/(spam + ham)
    pham = ham / (spam + ham)
    #print("P_Spam:", pspam)
    #print("P_Ham:", pham)
    fin = open(test, "r")
    textline = fin.readline() 
    
    while textline != "":
        is_spam = int(textline[:1]) 
        
        if is_spam == 1:
            testspam = testspam + 1 
        
        else:
            testham = testham + 1
            
        textline = cleantext(textline[1:])   
        words = remove_stop_words(textline, stop)
        #print(words)
        p1 , p2 = checkvocab(words , train_vocab)
        #print("P1:", p1)
        #print("P2:", p2)
        predict = compute_probability(p1, p2, pspam, pham )
        #print("Predict:", predict)
        
        if predict > 0.5 and is_spam == 1:
            tp +=1
        elif predict > 0.5 and is_spam == 0:
            fp+=1
        elif predict <= 0.5 and is_spam == 1:
            fn+=1
        elif predict <= 0.5 and is_spam == 0:
            tn+=1
        
        textline = fin.readline()
    ptest_spam = testspam/(testspam + testham)
    ptest_ham = testham / (testspam + testham)
    
    #Printing Results on Console for labelled test set
    print("Total Spam emails in Test set: ", testspam)
    print("Total Ham emails in Test set: ", testham)   
    print('*'*30)
    print("Predictions by Spam Filter")
    print('*'*30)
    print("True Positives: {} | True Negatives: {} | False Positives: {} | False Negatives: {}".format(tp, tn, fp, fn))
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    accuracy = round(accuracy,3)
    precision = tp / (tp + fp)
    precision = round(precision,3)
    recall = tp / (tp + fn)
    recall = round(recall,3)
    f1_score = 2 * (1 / ((1 / precision) + (1 / recall)))
    f1_score = round(f1_score,3)
    print("Accuracy: {} [{} %] " .format(accuracy, accuracy*100))
    print("Precision: {} [{} %] ".format(precision, precision*100))
    print("Recall: {} [{} %] " .format(recall, recall*100))
    print("F1_score: {} [{} %] " .format(f1_score, f1_score*100))

    fin.close()

def compute_probability(p1 , p2, ps, ph): #p1 : P(SL/S) , p2:P(SL/notS) , ps: P(S) , ph: P(NotH)
    #print("p1: {} p2: {} ps: {} ph:{}"  .format(p1, p2, ps, ph))
    pf = (p1 * ps) / (( p1 * ps ) + ( p2 * ph ))
    
    return pf

def checkvocab(words, vocab):
    p1 = 1
    p2 = 1
    
    for key, values in vocab.items():
        if key in words:
            p1 += math.log(values[1])  #Used LOG to avoid underflow
            p2 += math.log(values[0])  #Used LOG to avoid underflow
        
        else: 
            p1 += math.log(1 - values[1])  #Used LOG to avoid underflow
            p2 += math.log(1 - values[0])   #Used LOG to avoid underflow
            
    return math.exp(p1), math.exp(p2)
    
    
def main():
    #train = "Geastrain.txt"
    #stop = "StopWords.txt"
    #test = "Geastest.txt"
    
    #Take user inputs for train and stop text files
    train = input("Enter name of training set file:")
    stop = input("Enter name of stop words file:")
        
    #Create Train Vocabulary    
    train_vocab, spam, ham = vocabulary(train,stop)
    
    #Take users input for labelled test data to predict
    test = input("Enter name of test words file:")
    
    print("\n")
    #Testing on labelled test set and getting performance metrics
    testvocab(test,stop, train_vocab, spam, ham)
        
     
main()
    
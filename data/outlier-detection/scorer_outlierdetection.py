# -*- coding: utf-8 -*-

### Author: Jose Camacho Collados

import os
import fileinput
from math import sqrt
import operator
import sys

class OutlierDetectionCluster:
    #Class modeling a cluster of the dataset, composed of its topic name, its corresponding elements and the outliers to be detected 
    def __init__(self,elements,outliers,topic=""):
        self.elements=elements
        self.outliers=outliers
        self.topic=topic

class OutlierDetectionDataset:
    #Class modeling a whole outlier detection dataset composed of various topics or clusters
    def __init__(self,path):
        self.path=path
        self.setWords=set()
        self.clusters=set()
    def readDataset(self):
        print ("\nReading outlier detection dataset...")
        dict_cluster_elements={}
        dict_cluster_outliers={}
        listing=os.listdir(self.path)
        for in_file in listing:
            if in_file.endswith(".txt"):
                cluster_file=open(path_dataset+in_file).readlines()
                cluster_name=in_file.replace(".txt","")
                set_elements=set()
                set_outliers=set()
                cluster_boolean=True
                for line in cluster_file:
                    if cluster_boolean:
                        if line!="\n":
                            word=line.strip().decode('utf-8').replace(" ","_")
                            set_elements.add(word)
                            self.setWords.add(word)
                            if "_" in word:
                                for unigram in word.split("_"):
                                    self.setWords.add(unigram)
                        else: cluster_boolean=False  
                    else:
                        if line!="\n":
                            word=line.strip().decode('utf-8').replace(" ","_")
                            set_outliers.add(word)
                            self.setWords.add(word)
                            if "_" in word:
                                for unigram in word.split("_"):
                                    self.setWords.add(unigram)
                self.clusters.add(OutlierDetectionCluster(set_elements,set_outliers,cluster_name))
                
def boolean_answer(answer):
    if answer.lower()=="y" or answer.lower()=="yes": return True
    elif answer.lower()=="n" or answer.lower()=="no": return False
    else:
        new_answer=raw_input('Please answer "Yes" or "No"')
        return boolean_answer(new_answer)
        
    
def module(vector):
    #Module of a vector
    suma=0.0
    for dimension in vector:
        suma+=dimension*dimension
    return sqrt(suma)

def scalar_prod(vector1,vector2):
    #Scalar product between two vectors
    prod=0.0
    for i in range(len(vector1)):
        dimension_1=vector1[i]
        dimension_2=vector2[i]
        prod+=dimension_1*dimension_2
    return prod

def cosine(vector1,vector2):
    #Cosine similarity between two vectors
    module_vector_1=module(vector1)
    if module_vector_1==0.0: return 0.0
    module_vector_2=module(vector2)
    if module_vector_2==0.0: return 0.0
    return scalar_prod(vector1,vector2)/(module(vector1)*module(vector2))

def pairwisesimilarities_cluster(setElementsCluster,input_vectors):
    #This function calculates all pair-wise similarities between the elements of a cluster and stores them in a dictionary
    dict_sim={}
    for element_1 in setElementsCluster:
        for element_2 in setElementsCluster:
            if element_1!=element_2:
                dict_sim[element_1+" "+element_2]=cosine(input_vectors[element_1],input_vectors[element_2])
    return dict_sim

def compose_vectors_multiword(multiword,input_vectors,dimensions):
    #Given an OOV word as input, this function either returns a vector by averaging the vectors of each token composing a multiword expression or a zero vector
    vector_multiword=[0.0]*dimensions
    cont_unigram_in_vectors=0
    for unigram in multiword.split("_"):
        if unigram in input_vectors:
            cont_unigram_in_vectors+=1
            vector_unigram=input_vectors[unigram]
            for i in range(dimensions):
                vector_multiword[i]+=vector_unigram[i]
    if cont_unigram_in_vectors>0:
        for j in range(dimensions):
            vector_multiword[j]=vector_multiword[j]/cont_unigram_in_vectors
    return vector_multiword

def getting_vectors(path_vectors,set_words):
    #Reads input vectors file and stores the vectors of the words occurring in the dataset in a dictionary
    print ("Loading word vectors...")
    dimensions=-1
    vectors={}
    vectors_file=fileinput.FileInput(path_vectors)
    for line in vectors_file:
        word=line.split(" ",1)[0].decode('utf-8')
        if word in set_words:
            linesplit=line.strip().split(" ")
            if dimensions!=len(linesplit)-1:
                if dimensions==-1: dimensions=len(linesplit)-1
                else: print ("WARNING! One line with a different number of dimensions")
            vectors[word]=[]
            for i in range(dimensions):
                vectors[word].append(float(linesplit[i+1]))
    print ("Number of vector dimensions: "+str(dimensions))
    for word in set_words:
        if word not in vectors:
            vectors[word]=compose_vectors_multiword(word,vectors,dimensions)
    print ("Vectors already loaded")
    return vectors,dimensions

def main(path_dataset, path_vectors):

    dataset=OutlierDetectionDataset(path_dataset)
    dataset.readDataset()
    input_vectors,dimensions=getting_vectors(path_vectors,dataset.setWords)

    dictCompactness={}
    countTotalOutliers=0
    numOutliersDetected=0
    sumPositionsPercentage=0
    detailedResultsString=""
    resultsByClusterString=""
    for cluster in dataset.clusters:
        resultsByClusterString+="\n\n -- "+cluster.topic.encode('utf-8')+" --"
        detailedResultsString+="\n\n -- "+cluster.topic.encode('utf-8')+" --\n"
        dictSim=pairwisesimilarities_cluster(cluster.elements,input_vectors)
        numOutliersDetectedCluster=0
        sumPositionsCluster=0
        countTotalOutliers+=len(cluster.outliers)
        for outlier in cluster.outliers:
            compScoreOutlier=0.0
            dictCompactness.clear()
            for element_cluster_1 in cluster.elements:
                sim_outlier_element=cosine(input_vectors[element_cluster_1],input_vectors[outlier])
                compScoreElement=sim_outlier_element
                compScoreOutlier+=sim_outlier_element
                for element_cluster_2 in cluster.elements:
                    if element_cluster_1!=element_cluster_2:
                        compScoreElement+=dictSim[element_cluster_1+" "+element_cluster_2]
                dictCompactness[element_cluster_1]=compScoreElement
                detailedResultsString+="\nP-compactness "+element_cluster_1.encode('utf-8')+" : "+str(compScoreElement/len(cluster.elements))
            dictCompactness[outlier]=compScoreOutlier
            detailedResultsString+="\nP-compactness "+outlier.encode('utf-8')+" : "+str(compScoreOutlier/len(cluster.elements))
            sortedListCompactness=(sorted(dictCompactness.iteritems(), key=operator.itemgetter(1),reverse=True))
            position=0
            for element_score in sortedListCompactness:
                element=element_score[0]
                if element==outlier:
                    sumPositionsCluster+=position
                    if position==8: numOutliersDetectedCluster+=1
                    break
                position+=1
            detailedResultsString+="\nPosition outlier "+outlier.encode('utf-8')+" : "+str(position)+"/"+str(len(cluster.elements))+"\n"
            
        numOutliersDetected+=numOutliersDetectedCluster
        sumPositionsPercentage+=(sumPositionsCluster*1.0)/len(cluster.elements)
        scoreOPP_Cluster=(((sumPositionsCluster*1.0)/len(cluster.elements))/len(cluster.outliers))*100
        accuracyCluster=((numOutliersDetectedCluster*1.0)/countTotalOutliers)*100.0
        resultsByClusterString+="\nAverage outlier position in this topic: "+str(scoreOPP_Cluster)
        resultsByClusterString+="\nOutliers detected percentage in this topic: "+str(accuracyCluster)
        resultsByClusterString+="\nNumber of outliers in this topic: "+str(len(cluster.outliers))

    scoreOPP=((sumPositionsPercentage*1.0)/countTotalOutliers)*100
    accuracy=((numOutliersDetected*1.0)/countTotalOutliers)*100.0
    print ("\n\n ---- OVERALL RESULTS ----\n")
    print ("OPP score: "+str(scoreOPP))
    print ("Accuracy: "+str(accuracy))
    print ("\nTotal number of outliers: "+str(countTotalOutliers))

    answer=raw_input("\n\nWould you like to see the results by topic? [Y|N]")
    boolean=boolean_answer(answer)
    if boolean==True:
        print (resultsByClusterString)
        answer_2=raw_input("\n\nWould you like to see a more detailed summary? [Y|N]")
        boolean_2=boolean_answer(answer_2)
        if boolean_2==True:
            print (detailedResultsString)

if __name__ == '__main__':

    args = sys.argv[1:]

    if len(args) == 2:

        path_dataset = args[0]
        path_vectors = args[1]

        main(path_dataset, path_vectors)

    else:
        sys.exit('''
            Requires:
            path_dataset -> Path of the outlier detection directory
            path_vectors -> Path of the input word vectors
            ''')


        
                    
            
            



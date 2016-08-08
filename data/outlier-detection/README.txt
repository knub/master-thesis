=================================================================================================================================
					Find the word that does not belong:
			A Framework for an Intrinsic Evaluation of Word Vector Representations

				     José Camacho Collados and Roberto Navigli
=================================================================================================================================


This package contains five files and a directory:


* 8-8-8_Dataset/ (Directory): It contains eight different topics, each of them composed by a cluster of eight elements and
		              eight outliers which do not belong to the given topic. For more information about the format of
			      the dataset, please see the guidelines given to the annotators ("guidelines_cluster_creation.txt").

* README.txt (this file)

* ACL16_REPEVAL_Outlier_Detection.pdf (Reference paper): Please read this paper to have a more detailed information about the 
						         outlier detection task.

* guidelines_cluster_creation.txt : Guidelines given to the annotators to create the 8-8-8 outlier detection dataset.

* sample_skipgram_wikipedia_vectors.txt : Small sample of the vectors calculated by using the Skip-Gram model of Word2Vec on the
					  Wikipedia corpus (dump of November 2014).

* scorer_outlierdetection.py : Python script to test word vectors on the outlier detection dataset (requires Python 2). Find 
			       below more information on how to use it.

=================================================================================================================================
INSTRUCTION FOR EVALUATING WORD VECTORS ON THE OUTLIER DETECTION TASK
=================================================================================================================================

Input: 
	- Word vectors file (e.g. sample_skipgram_wikipedia_vectors.txt). The format of the vectors should follow the standard of
	 "sample_skipgram_wikipedia_vectors.txt": a word per line and the dimensions should be space-separated. 
	
	- Outlier detection dataset directory (e.g. 8-8-8_Dataset/). If needed, more clusters can be added to the directory by 
	  following the same format indicated in "guidelines_cluster_creation.txt"

Output: Short summary of results of the word vectors on the task (accuracy and OPP measures). 
	After this short summary of results is printed on the screen, the program will ask the user if he/she would like to have 
	the results divided by cluster and later if he/she would like to have a more detailed summary of the results.

-IMPORTANT NOTE-: The code can takes as input multiword expressions vectors (e.g. Mercedes_Benz). In the case a multiword 
		expression is Out-Of-Vocabulary (OOV), by default a compositional function ("compose_vectors_multiword") based on 
		averaging its unigram vectors (e.g. Mercedes and Benz) is applied. Whenever a word is OOV or in the case of a OOV 
		multiword whose unigram components are also OOV, the program takes as default the zero vector (i.e. all its 
		dimensions are zero).


Intructions to run the Python script "scorer_outlierdetection.py": 

The code takes the following parameters: path_dataset, path_vectors
	path_dataset	: Path of the outlier detection directory.
        path_vectors	: Path of the input word vectors.

Run it in the terminal by typing the following expression: 
	$ python scorer_outlierdetection.py path_dataset path_vectors

Example of usage: 
	$ python scorer_outlierdetection.py 8-8-8_Dataset/ sample_skipgram_wikipedia_vectors.txt

If you run this code as in the example, the output should be as follows:

-----------------------------------
Reading outlier detection dataset...
Getting word vectors...
Number of vector dimensions: 300
Vectors already loaded


 ---- OVERALL RESULTS ----

OPP score: 93.75
Accuracy: 70.3125

Total number of outliers: 64

Would you like to see the results by topic? [Y|N]
-----------------------------------

NOTE: In the detailed summary of the resuls, P-compactness score refers to the pseudo-inverted compactness score mentioned
      in the reference paper.	

=================================================================================================================================
REFERENCE PAPER
=================================================================================================================================

When using these resources, please refer to the following paper (included in the package as "ACL16_REPEVAL_Outlier_Detection.pdf"):

	José Camacho-Collados and Roberto Navigli. 
	Find the word that does not belong: A Framework for an Intrinsic Evaluation of Word Vector Representations
	Proceedings of the  ACL Workshop on Evaluating Vector Space Representations for NLP, 
	Berlin, Germany, August 12, 2016. 


=================================================================================================================================
CONTACT
=================================================================================================================================
 
If you have any enquiries about any of the resources, please contact José Camacho Collados (collados [at] di.uniroma1 [dot] it)

=================================================================================================================================


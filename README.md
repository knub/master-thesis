# Improving Topic Models using Word Embeddings

A topic model is a probabilistic model, which tries to discover abstract topics, that occur in a large document collection.
Topic models group words together into one topic, when the words co-occur in many documents.
For example, here are five topics, that are discovered in the English Wikipedia:

1. ``population economic million government country economy world percent development growth``	
2. ``court law case legal supreme justice states judge act rights``
3. ``book published books literature history literary work wrote works poetry``
4. ``teams cup league tournament round season team championship group football``
5. ``tax price money market financial income value rate pay economic``

Topic models are able to assign topic proportions to a given document.
So the topic model could tell that document about the FIFA scandal is ``25% topic 1 + 50% topic 4 + 25% topic 5``.
Topic models are helpful for analyzing large document collections, detecting trends and changes and can also be used for recommendations (*show me articles similar to the one I am currently reading*).

In recent years, a different technique named word embeddings have gained popularity in the natural language processing community.
Word embedding is a technique, which assigns words a point in a high-dimensional vector space (*the words are embedded in the space*).
Using a clever learning strategy, words which are similar to each other are placed together in the vector space.
Also, the vector space encodes semantic relationships between words.
For example, the vector which is closest to ``king - man + woman`` is the vector of the word ``queen``.

In traditional topic modelling, each word is treated as an atomic unit, and no relationship between words exist a priori.
All the information about the topics is extracted from co-occurrence counts.
Using word embeddings, it is possible to encode semantic similarity between words a priori.
In this master thesis, I will improve the existing topic modeling approaches by incorporating word embeddings in the model, ultimately leading to more useful topic models.

## Status

I started working on my master thesis in June 2016 and will finish by the end of the year.

## Folder structure

* ``code``: includes Python and Scala code for data processing, models and my experiments
* ``data``: contains evaluation data sets
* ``expose``: contains the expose (first draft about the thesis at the beginning of the project)
* ``notebooks``: contains Jupyter notebooks I use for analyzing and visualizing experiments
* ``presentation``: contains images used in presentations about my thesis
* ``thesis``: contains the thesis text
* ``webapp``: contains the webapp used for the word intrusion study

## Contact

Feel free to contact me, if you have any questions about the thesis.
My name is Stefan Bunk and you can write me an e-mail to ``firstname.lastname@student.hpi.uni-potsdam.de``.

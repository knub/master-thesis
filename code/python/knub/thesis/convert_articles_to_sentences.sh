#!/bin/bash

for articles_split in /data/wikipedia/2016-06-21/articles.txt.split.*; do
	match="articles"
	replacement="sentences"
	sentences_split=${articles_split/$match/$replacement}
	echo $sentences_split
	python sentence_tokenization.py $articles_split $sentences_split &
done

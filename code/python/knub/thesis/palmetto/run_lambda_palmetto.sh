#!/bin/bash

for i in 2 3 4 5 6 7 8 9
do
	echo $i
	python run_palmetto.py /data/wikipedia/2016-06-21/topic-models/topic.20news.50-1500.with-classes/model.google.model.welda.lambda-0-$i.*.topics
done

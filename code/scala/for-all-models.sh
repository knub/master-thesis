#!/bin/bash
# Exit on first failure
trap 'exit' ERR

for filename in /data/wikipedia/2016-06-21/topic-models/*.model; do
      model_name=$(basename $filename)
      log_name="$model_name.log"
      echo $model_name
      bin/topic-model topic-model --model-file-name $filename | tee $log_name
done

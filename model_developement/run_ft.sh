#!/bin/bash

# Check if argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_name> -> [scifact-open, contract-nli, evidenceinference, ildc] $1 <batch_size> $2 <epoch> $3 <learning_rate> $4 <save_dir> $5"
    exit 1
fi

TASK_NAME="$1"


if [ $# -lt 2 ]; then
    BATCH=1
else
    BATCH="$2"
fi


if [ $# -lt 3 ]; then
    EPOCH=10
else
    EPOCH="$3"
fi


if [ $# -lt 4 ]; then
    if [ "$TASK_NAME" = "scifact-open" ]; then
        LR=0.00005
    else if [ "$TASK_NAME" = "contract-nli" ]; then
        LR=0.00001
    else if [ "$TASK_NAME" = "evidenceinference" ]; then
        LR=0.000003
    else if [ "$TASK_NAME" = "ildc" ]; then
        LR=0.000005
    fi
else
    LR="$4"
fi


if [ $# -lt 5 ]; then
    Save_dir="./output_model/"$TASK_NAME"/"
else
    Save_dir="$5"
fi


if [ "$TASK_NAME" = "scifact-open" ]; then
    truncation_side="right"
else
    truncation_side="left"
fi

File_Src="./data/"$TASK_NAME"/"

MODEL='google/flan-t5-xl'
# MODEL='google/flan-t5-small'    ### To test and ensur everything works fine

echo "---> Running for" $TASK_NAME "task with batch_size: " $BATCH "epoch: " $EPOCH "learning rate: " $LR
echo "---> MODEL: "$MODEL;
echo "---> Saving the model to " $Save_dir
echo "_____________________________________________________________________________________________________________________"


time python3 src/classify.py \
  --model_name_or_path $MODEL \
  --text_column "input" \
  --answer_column "choice" \
  --task_name $TASK_NAME \
  --per_device_eval_batch_size $BATCH \
  --per_device_train_batch_size $BATCH \
  --do_train \
  --do_eval \
  --do_predict \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --truncation_side $truncation_side \
  --train_file $File_Src'train.json' \
  --validation_file $File_Src'dev.json' \
  --test_file $File_Src'test.json' \
  --max_source_length 4200 \
  --eval_steps 100 \
  --save_steps 100 \
  --logging_steps 100 \
  --save_strategy steps \
  --evaluation_strategy steps \
  --load_best_model_at_end \
  --gradient_checkpointing \
  --gradient_accumulation_steps 64 \
  --predict_with_generate \
  --overwrite_output_dir \
  --output_dir $Save_dir


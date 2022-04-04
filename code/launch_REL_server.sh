#!/bin/bash

export PATH=/nfs/yding4/conda_envs/rel/bin:$PATH
export LD_LIBRARY_PATH=/nfs/yding4/conda_envs/rel/lib:$LD_LIBRARY_PATH

CODE=/nfs/yding4/REL/code/server_with_sentence2ner.py

NER_MODEL='/nfs/yding4/REL/data/first_trained_model/final-model.pt'
#NER_MODEL='ner-fast'

export CUDA_VISIBLE_DEVICES=''
python ${CODE}  \
    --ner-model ${NER_MODEL}    \
    --sentence2ner-file '/nfs/yding4/REL/data/sentence2ner/sample_sentence2ner.jsonl'


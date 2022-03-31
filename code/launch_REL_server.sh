#!/bin/bash

export PATH=/nfs/yding4/conda_envs/rel/bin:$PATH
export LD_LIBRARY_PATH=/nfs/yding4/conda_envs/rel/lib:$LD_LIBRARY_PATH

CODE=/nfs/yding4/REL/code/launch_REL_server.py

export CUDA_VISIBLE_DEVICES=''
python ${CODE}


#!/bin/bash
MEM=1024
N_GPUS=1

model load python_gpu/3.6.4

echo "running classify model"
bsub -n 1 -W 4:00 -R "rusage[mem=$MEM,ngpus_excl_p=$N_GPUS]" python run.py
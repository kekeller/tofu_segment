#!/bin/bash
MEM=20480
N_GPUS=1

echo "running classify model"
bsub -n 1 -W 4:00 -R "rusage[mem=$MEM,ngpus_excl_p=$N_GPUS]" python classify.py

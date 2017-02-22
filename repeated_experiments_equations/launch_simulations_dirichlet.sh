#!/bin/bash

source ~/miniconda2/bin/activate my-rdkit-env
export OPENBLAS_NUM_THREADS=1
for i in 3
do
    echo $i
    cd simulation$i/character/
    temp_folder=`mktemp`
    rm $temp_folder
    mkdir $temp_folder
    echo $temp_folder
    export THEANO_FLAGS=base_compiledir=$temp_folder
    nohup python  run_experiment.py --exp_seed=$i &
    cd ../..
done

#!/bin/bash

for i in {1..10}
do
    echo $i
    rm -rf simulation$i
    mkdir simulation$i
    cp -r source_files/character simulation$i
    cp -r source_files/grammar simulation$i
done

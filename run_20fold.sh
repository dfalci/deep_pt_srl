#!/bin/bash
for (( i=1; i<=20;i++))
do
	python train.py $i
done

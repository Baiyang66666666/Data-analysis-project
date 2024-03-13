#!/bin/bash
#$ -l h_rt=6:00:00  #time needed
#$ -pe smp 10 #number of cores
#$ -l rmem=20G #number of memory
#$ -P rse-com6012 # require a com6012-reserved node
#$ -q rse-com6012.q # specify com6012 queue
#$ -o /home/acp22bq/com6012/ScalableML/Output/Q5/Q5_output.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M bqu5@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 20g --executor-memory 20g /home/acp22bq/com6012/ScalableML/Code/Q5_code.py

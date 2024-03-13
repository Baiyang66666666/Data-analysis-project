#!/bin/bash
#$ -l h_rt=6:00:00  # time needed in hours:mins:secs
#$ -pe smp 2 # number of cores requested
#$ -l rmem=8G # size of memory requested
#$ -o /home/acp22bq/com6012/ScalableML/Output/Q1_output.txt  # This is where your output and errors are logged
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M acp22bq@shef.ac.uk # notify you by email, remove this line if you don't want to be notified
#$ -m ea # email you when it finished or aborted
#$ -cwd # run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit /home/acp22bq/com6012/ScalableML/Code/Q1_code.py  # .. is a relative path, meaning one level up
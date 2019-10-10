#!/bin/bash
#PBS -P u46
#PBS -qnormal
#PBS -N rforest_brindi
#PBS -m ae
#PBS -M cate.kooymans@ga.gov.au
#PBS -l wd
#PBS -l walltime=12:00:00
#PBS -l mem=128GB,ncpus=1,jobfs=4000MB
#PBS -j oe

module purge
module use /g/data/v10/public/modules/modulefiles
module load dea-prod
module add agdc_statistics
module list

python my_scripts/RFC_tree_calss_Brin.py


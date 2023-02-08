#!/bin/bash
# Get rankings filename from command line argument -f
while getopts f: option
do
case "${option}"
in
f) FILENAME=${OPTARG};;
esac
done

# Submit rankings
kaggle competitions submit -c expedia-personalized-sort -f $FILENAME -m "Submission using kaggle API"
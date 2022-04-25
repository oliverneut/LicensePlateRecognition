#!/bin/bash


T=$(find /home/imageprocessingcourse/ -type f -name "*test*" | head -n 1)

/usr/local/bin/python /home/imageprocessingcourse/main.py --file_path $T --output_path ./Output.csv

G=$(find /home/imageprocessingcourse/ -type f -name "*groundTruth*" | head -n 1)

/usr/local/bin/python /home/imageprocessingcourse/evaluation.py --file_path ./Output.csv --ground_truth_path $G

#Comment out lines until here if you dont want docker to evaluate on start.

#Do not ever comment out below line
/bin/bash




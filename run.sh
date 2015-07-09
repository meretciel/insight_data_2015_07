#!/usr/bin/env bash

# example of the run script for running the word count

# I'll execute my programs, with the input directory tweet_input and output the files in the directory tweet_output


# Instructoin and example of running the program
# Note that the order is important. 

# python my_program.py ./tweet_input/my_input.txt ./tweet_output/output_for_word_count.txt ./tweet_output/output_for_medians.txt

# The python code will read the input from ./tweet_intput/my_input.txt. (./tweet_intput/ directory is prefixed).
# The python code will save the output to the indicated directory

# Examples:

# Run simple_approach.py
# python ./src/simple_approach.py ./tweet_input/tweets.txt ./tweet_output/f1.txt ./tweet_output/f2.txt

# Run Spark version
# python ./src/spark_version.py  ./tweet_input/tweets.txt ./tweet_output/f1.txt ./tweet_output/f2.txt





python ./src/spark_version.py ./tweet_input/tweets.txt ./tweet_output/f300.txt ./tweet_output/f301.txt



Insight Data Project Report
======================





##Introduction


The main purpose of this project is to implement two features mentioned in the project requirement:


1.  Calculate the total number of times each word has been tweeted.
2.  Calculate the median number of unique words per tweet, and update this median as tweets come in.

There are difficulties here:

1.  The size of the input data is enormous. Tweeter claims that they generate 12GB text data per day.
2.  We need an efficient algorithm to calculate the rolling median of streaming integers.


For the first problem, we do not have many choices because at the end we still need to iterate through the whole input file at lease once. One possible solution is to split input files and follow the MapReduce programming model or we can use framework such as Spark directly.

For the second problem, here are some thoughts. In a general context, it takes O(n) to compute the median of a unsorted array. However, as we know one tweet can only have 140 characters, hence 70 words at most. So our problem becomes how to compute the rolling median of streaming integers with known upper bound. If approximate results are acceptable, we can use P^2 algorithm do dynamically calculate the rolling median. Both of the algorithms have O(1) complexity.


## Execution Environment

###### Import statements for Python
```
import numpy as np
import matplotlib.pyplot as plt
from string import split
from numpy.random import randint
import os
import sys
import time
```

###### Spark
We need to have Spark installed on the machine. It is better to set `log4j.rootCategory=ERROR, console` in /spark/conf/log4j.properties so that we can save some time from printing all the log information.





##Simple Approach

In this section, we will provide one straight forward solution to our problem. The reason  we start from a simple solution is that we want to investigate in different parts of the problem.

##### Word Count
To compute the word count, we will create a large dictionary. The key is the word and the value is the number of that word in the input files. Every time we read a word, if it is already in the dictionary, we increment the value by one, otherwise we add this new word to the dictionary and set the value equal to one.

##### Median Number of unique words per tweet
To compute the rolling median, we will provide two different method. The first one will compute the exact median while the second one will calculate the approximate number.

###### First method  

As we mentioned in the introduction, if we assume that there are at most 70 words per tweet, then the calculation of the median becomes much easier. The general idea is to create a kind of histogram of different number of words and then find the number that sits in the middle. Let counters denote the histogram, for example counters[10] is the number of tweets that has 10 words. To find the median, we just select the number N such that sum(counters[:N]) == 0.5 * total_number_of_tweets


###### Second method  

The second method we want to introduce is  [P2 algorithm] (http://pierrechainais.ec-lille.fr/Centrale/Option_DAD/IMPACT_files/Dynamic%20quantiles%20calcultation%20-%20P2%20Algorythm.pdf). This is a heuristic algorithm and will calculate the approximate median. The complexity of the algorithm is O(1) and it has a quite a good performance when the sample size is large. The figure below shows the performance of this algorithm.


![]( ../images/approximate_median_1.png)

###### Coparison

size 	|	first method		| second method
---		|	:---:					|	:---:
8500	| 0.1301s				| 0.0852s
20000  	| 0.2996s				| 0.1864s

We would say P2 algorithm is quite efficient. However note that we can optimize the first method. Essentially, the first method calculates the exact 0.5-quantile for every new coming element. In order to optimize, we can memorize those values around 0.5-quantile and in that way we do not need re-calculate the 0.5-quantile every time.
 


## Performance Analaysis

Here we use the line-by-line profiler to get a general idea of the performance. It takes 17% of time to read and process input file; 39% of time to do the word counts;  18% of time to compute the rolling median and 23% of time to write results to disk. 

Note that except computing the rolling median, all other parts can be done in a parallel way and they make up 79% of processing time. This observation confirm us that a parallel framework is needed to solve this problem. 

Of course, we can manually create some parallel features with the help of multiprocessing module and collections module. In particular, we would be interested in `multiprocessing.Pool.map` and `collections.defaultdict`. However, we decide to use the Spark directly.


```

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    49                                           @profile
    50                                           def main():
    51         1           12     12.0      0.0      counters = np.zeros(_MAX_LENGTH, dtype=int)
    52         1            1      1.0      0.0      list_rolling_medians = []
    53         1            2      2.0      0.0      dict_aggregate = dict()
    54                                           
    55         1        44465  44465.0      0.2      with open(r'../tweet_input/sample_tweets_3.txt', 'r') as in_file :
    56                                           
    57    100001       157134      1.6      0.9          for idx, line in enumerate(in_file):
    58                                                       # split the line
    59                                                       # given the context of the project, the separator is space
    60    100000       482979      4.8      2.7              list_of_words = split(line, sep=' ')
    61                                           
    62                                                       # process the list_of_words
    63                                                       # the goal is to create a dictionary of words,
    64                                                       # the key is the word itself and the value is the word frequency in the line
    65    100000       186830      1.9      1.0              _set = set()
    66   3542632      2484409      0.7     13.6              for word in list_of_words:
    67                                                           # update _dict
    68   3442632      2933926      0.9     16.1                  _set.add(word)
    69                                           
    70                                                           # update the aggregate dictionary
    71   3442632      4125860      1.2     22.7                  dict_aggregate[word] = dict_aggregate.get(word, 0) + 1
    72                                           
    73                                           
    74                                           
    75                                                       # add two attributes: dictionary id and the number of unique words in the lines
    76    100000        87106      0.9      0.5              num_of_unique_words = len(_set)
    77                                           
    78                                           
    79                                                       # update the counter
    80    100000       152628      1.5      0.8              counters[num_of_unique_words] += 1
    81    100000      3222244     32.2     17.7              current_median = compute_rolling_median_from_counters(counters, idx+1)
    82    100000        94951      0.9      0.5              list_rolling_medians.append(current_median)
    83         1            2      2.0      0.0      in_file.close()
    84                                           
    85                                           
    86         1         1265   1265.0      0.0      with open(r'../tweet_output/output_simple_version_1.csv', 'w') as out_file:
    87   1058498      2270315      2.1     12.5          for key in sorted(dict_aggregate.keys()):
    88   1058497      1962756      1.9     10.8              out_file.write(str(key) + '    ' + str(dict_aggregate[key]) + '\n')
    89                                           
    90         1            5      5.0      0.0      out_file.close()

```

## Using Spark

In this section, we will introduce how we solve the problem with the help of Spark. 

As we mentioned previously, the problem can be divined into two sub-problem: word count problem and calculation of rolling median. For word count problem, the solution is straight forward. Following the MapReduce frame, we first split each tweet to individual words, then map each word to (word, 1) and finally reduce the paris by key. Translated into python code:

```python
_file = sc.textFile('../tweet_input/tweets.txt')
counts = _file.flatMap(lambda line: line.split(' ')) \
                .map(lambda word : (word, 1)) \
                .reduceByKey(lambda a,b: a + b) \
                .sortByKey(True) \

```

To calculate the median number of unique words per tweet we need to first recored the number of unique words in each tweet. One way to to do that is to create a dictionary for each tweet, the key is the word and the value if the number of that word in the tweet and the number of the unique words is just the number of the keys in the dictionary. 

We will store the number of unique words in the dictionary and RDD as well, the key for this item is a tuple named ('marked',).  With this setting, we can easily select the number of unique words from the RDD later on. Here is the code of the function convert_line_to_dict, it will take each line in tweets.txt as input and return (the times of) the dictionary mentioned above .


```python
def convert_line_to_dict(line):
    list_of_words = split(line[:-1], sep=' ')

    _dict = dict()

    for word in list_of_words:
        _dict[word] = _dict.get(word, 0) + 1

    num_of_unique_words = len(_dict)

    _dict[('marked', )] = num_of_unique_words

    return _dict.items()
    
```

Finally, to separate the word counts and the number of unique words, we apply filter to RDD:

```
counts = processed.filter(lambda _tuple: not isinstance(_tuple[0], tuple)) 
# ...
num_of_words = processed.filter(lambda _tuple: isinstance(_tuple[0], tuple))
```

##### Performance

We tested the Spark standalone application on local machine with 4 cores (probably 1 master and 3 workers). We generate two sample input files of different size, one is 23M and the other is 1.6G. It turns out the the spark application runs much slower than the regular python program.  

Here are some thoughts of the performance issue:

1. Spark does extra work such as split the file, generate new RDD, establish communication. Therefore, we might need a larger test case to see the benefits of using spark.

2. It seems that it takes a significant amount of time to process saveAsTextFile function. For example, when testing with 23M input file, if we do not save our results, it takes 23 seconds to run the program; however, if we call saveAsTestFile, it takes 42 seconds to finish the program.


We tried different configurations on our local machine but there was no significant improvement of the performance. We need to test the program on the cluster for further discussion.







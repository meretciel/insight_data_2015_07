import numpy as np
from string import split
import time

import subprocess
from apprMedian import compute_approximate_rolling_median_from_list
from calculateMedian import compute_rolling_median_from_list, compute_median_from_counters


import sys
import os
os.environ['SPARK_HOME'] = '/Users/Ruikun/Downloads/spark-1.4.0-bin-hadoop2.6'
os.environ['HADOOP_HOME'] = '/Users/Ruikun/Downloads/hadoop-2.7.0'
sys.path.append('/Users/Ruikun/Downloads/spark-1.4.0-bin-hadoop2.6/')
sys.path.append('/Users/Ruikun/Downloads/spark-1.4.0-bin-hadoop2.6/python')
sys.path.append('/Users/Ruikun/Downloads/spark-1.4.0-bin-hadoop2.6/python/lib')
sys.path.append('/Users/Ruikun/Downloads/spark-1.4.0-bin-hadoop2.6/python/lib/py4j-0.8.2.1-src.zip')
sys.path.append('/Users/Ruikun/Downloads/spark-1.4.0-bin-hadoop2.6/python/spark')



# we make the assumption that one tweet can only
# have 140 characters, hence 70 words
_MAX_LENGTH = 70
_PATH = r'./tweet_output/spark_output.csv'


#from pyspark import SparkContext, SparkConf
from pyspark import SparkContext
from pyspark import SparkConf

conf = SparkConf().setAppName('insight_data_project').setMaster('local[4]') \
        .set('spark.python.worker.memeory', '256m')
sc = SparkContext(conf=conf)


def convert_line_to_dict(line):
    list_of_words = split(line[:-1], sep=' ')

    _dict = dict()

    for word in list_of_words:
        _dict[word] = _dict.get(word, 0) + 1

    num_of_unique_words = len(_dict)

    _dict[('marked', )] = num_of_unique_words

    return _dict.items()





def main():
    print 'Spark is running.'
    _file = sc.textFile(sys.argv[1], 32)
    processed = _file.flatMap(convert_line_to_dict)

    #processed.cache()

    counts = processed.filter(lambda _tuple: not isinstance(_tuple[0], tuple)) \
                        .reduceByKey(lambda a,b: a + b) \
                        .sortByKey(True) \
                        .map(lambda _tuple: str(_tuple[0]) + '    ' + str(_tuple[1]))

    num_of_words = processed.filter(lambda _tuple: isinstance(_tuple[0], tuple)) \
                            .map(lambda _tuple : _tuple[1]) \
                            .collect()




    # process the output files
    # Spark will produce a folder that contains several files.
    # The files whose name starts with 'part' contain the actual output
    #
    # we will set coalesce(1), therefore there should be only one file start with 'part'

    print 'saving the output'
    counts.coalesce(1).saveAsTextFile(_PATH)

    current_path = os.getcwd()
    os.chdir(_PATH)

    output = subprocess.check_output('ls')
    list_of_files = split(output)
    filename = filter(lambda x: x.startswith('part'), list_of_files)[0]
    full_filename = os.path.join(_PATH, filename)

    # change to the original directory
    os.chdir(current_path)

    # move the file
    command = 'mv '  + full_filename + ' ' +  sys.argv[2]
    print 'executing {}'.format(command)
    os.system(command)



    list_rolling_medians = compute_rolling_median_from_list(num_of_words)

    with open(sys.argv[3], 'w') as outFile_median:
        for item in list_rolling_medians:
            outFile_median.write(str(item) + '\n')
    sc.stop()


if __name__ == '__main__':
    main()

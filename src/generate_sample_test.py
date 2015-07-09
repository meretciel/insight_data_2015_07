

import numpy as np
from numpy.random import randint


LIST_OF_CHARS = ['a','b','c','d','e','f','g','h','@','#','1','2','3','4','5','6','7', '8', '9', '0']

NUMBER_OF_LINES = 100000
#NUMBER_OF_LINES = 20
MAX_NUMBER = 1000000
random_ints = randint(0,MAX_NUMBER, randint(0,70))

def convert_number_to_word(n):
    list_chars =[]
    while n > 0:
        k = ((n % 10) * 23 ) % 20
        list_chars.append(LIST_OF_CHARS[k])
        n = n / 10

    return ''.join(list_chars)


def generate_one_line():
    random_ints = randint(0, MAX_NUMBER, randint(0,70))
    list_of_words = map(lambda x : convert_number_to_word(x), random_ints)
    return ' '.join(list_of_words)


if __name__ == '__main__':
    with open(r'../tweet_input/sample_tweets_1.txt', 'w') as _file:
        for i in xrange(NUMBER_OF_LINES):
            print '{} / {}'.format(i, NUMBER_OF_LINES)
            _file.write(generate_one_line())
            _file.write('\n')
    _file.close()
import numpy as np
import matplotlib.pyplot as plt
import time


_MAX_LENGTH = 70

def compute_median_from_counters(counters,len_arr):

    # calculate the current median
    if len_arr % 2 == 0:
        median_pos = len_arr / 2
    else:
        median_pos = len_arr / 2 + 1


    sum_of_count = 0
    number = 0
    while sum_of_count < median_pos:
        number += 1
        sum_of_count += counters[number]


    if len_arr % 2 == 1:            # if the number of elements is odd, we return the number directly
        return number
    elif sum_of_count> median_pos:  # if we have enough "number", then we can return this number
        return number
    else:                           # otherwise, we need to find the next number
        next_number = number + 1
        while counters[next_number] == 0:
            next_number += 1
        return (number + next_number) / 2.0



def compute_rolling_median_from_list(arr):
    counters = np.zeros(_MAX_LENGTH+1)
    rolling_median = []
    for idx, val in enumerate(arr):
        counters[val] += 1
        _median = compute_median_from_counters(counters, idx + 1)
        rolling_median.append(_median)
    return rolling_median


if __name__ == '__main__':
    arr = np.array(np.random.normal(1000,200,20000), dtype=int)
    arr = arr % (_MAX_LENGTH) + 1


    # compute the reference output
    res_ref = []
    for i in xrange(len(arr)):
        res_ref.append(np.median(arr[:(i+1)]))

    print 'calMedian.py'
    start = time.time()
    my_res = compute_rolling_median_from_list(arr)
    print time.time() - start

    # plot
    xx = xrange(len(arr))


    plt.plot(xx, res_ref, linestyle='-', color='red',  label='reference')
    plt.plot(xx, my_res,  linestyle='-', color='blue', label='P2')
    plt.legend(loc='upper right', prop={'size':8})
    plt.title('P2 algorithm test')
    plt.xlabel('sample size')
    plt.ylabel('median')
    plt.show()





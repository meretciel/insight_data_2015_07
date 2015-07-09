import numpy as np
import matplotlib.pyplot as plt
import time



_MAX_LENGTH  = 70


def apply_parabola_formula(q, n, q_plus_1, n_plus_1, q_minus_1, n_minus_1, d):
    _q = q + d / float(n_plus_1 - n_minus_1) * ( (n - n_minus_1 + d) * (q_plus_1 - q) / float(n_plus_1 - n) \
                                                 + (n_plus_1 - n - d) * (q - q_minus_1) / float(n - n_minus_1))
    _n = n + d

    return _q, _n

def apply_linear_formula(q, n, q_plus_1, n_plus_1, q_minus_1, n_minus_1, d):
    if d == 1:
        _q = q + (q_plus_1 - q) / float(n_plus_1 - n)
        _n = n + 1
    else:
        _q = q + (q_minus_1 - q) / float(n_minus_1 - n)
        _n = n - 1

    return _q, _n



def compute_approximate_rolling_median_from_list(arr):
    if len(arr) < 5:
        raise RuntimeError('Array should have at least 5 elements!')

    quantile = 0.5

    # compute the first five rolling medians
    rolling_median = np.zeros(len(arr))

    rolling_median[0] = arr[0]
    rolling_median[1] = 0.5 * (arr[0] + arr[1])
    rolling_median[2] = sorted(arr[:3])[1]

    tmp_sorted = sorted(arr[:4])
    rolling_median[3] = 0.5 * (tmp_sorted[1] + tmp_sorted[2])

    rolling_median[4] = sorted(arr[:5])[2]

    # setting the initial estimates
    desired_n = np.zeros(6)
    estimated_n = np.array([0,1,2,3,4,5])
    estimated_q = np.array([0] + list(sorted(arr[:5])))

    min_value = min(arr[:5])
    max_value = max(arr[:5])

    for idx_, val in enumerate(arr[5:]):
        idx = idx_ + 5

        # update min and max value and quantiles

        if val < min_value:
            min_value = val
            for i in [2,3,4,5]:
                estimated_n[i] += 1
            estimated_q[1] = min_value
        elif val > max_value:
            max_value = val
            estimated_q[5] = max_value
        else:
            for i in [2,3,4]:
                if estimated_q[i] > val:
                    estimated_n[i] += 1

        estimated_n[1] = 1
        estimated_n[5] = idx + 1

        # update the desired positions
        desired_n[1] = 1
        desired_n[2] = idx * quantile * 0.5 + 1
        desired_n[3] = idx * quantile + 1
        desired_n[4] = idx * (1 + quantile) * 0.5 + 1
        desired_n[5] = idx + 1

        # process
        for i in [2,3,4]:
            d = 0
            if desired_n[i] - estimated_n[i] > 1 and estimated_n[i+1] - estimated_n[i] > 1:
                d = 1
            elif desired_n[i] - estimated_n[i] < -1 and estimated_n[i-1] - estimated_n[i] < -1:
                d = -1

            if d != 0:
                q = estimated_q[i]
                n = estimated_n[i]
                q_plus_1 = estimated_q[i+1]
                n_plus_1 = estimated_n[i+1]
                q_minus_1 = estimated_q[i-1]
                n_minus_1 = estimated_n[i-1]

                # update the estimates
                tmp_q, tmp_n = apply_parabola_formula(q,n,q_plus_1,n_plus_1,q_minus_1,n_minus_1,d)

                # make sure that the marker heights is in a non-decrasing order
                if tmp_q < q_minus_1 or tmp_q > q_plus_1:
                    tmp_q, tmp_n = apply_linear_formula(q,n,q_plus_1,n_plus_1,q_minus_1,n_minus_1,d)

                if tmp_n > n_minus_1 and tmp_n < n_plus_1:
                    estimated_q[i], estimated_n[i] = tmp_q, tmp_n

        rolling_median[idx] = int(round(estimated_q[3] * _MAX_LENGTH))

    return rolling_median



if __name__ == '__main__':
    arr = np.array(np.random.normal(1000,200,20000), dtype=int)
    arr = arr % (_MAX_LENGTH) + 1


    # compute the reference output
    res_ref = []
    for i in xrange(len(arr)):
        res_ref.append(np.median(arr[:(i+1)]))

    print 'apprMedian.py'
    start = time.time()
    my_res = compute_approximate_rolling_median_from_list(arr / float(_MAX_LENGTH))
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




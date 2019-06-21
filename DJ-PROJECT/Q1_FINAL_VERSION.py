from __future__ import division
# Make // do the integer division, / the real division
from multiprocessing import Process

import sympy
import numpy as np
import re
import datetime


def calculate_ARp(p, Y, T):
    print('AR(%s)START' % p)
    a = sympy.symbols('a0:%s' % (p+1))
    sigma_square = sympy.symbols('sigma_square', real=True)
    likelihood_part = 0
    tmp = 0
    for t in range(p+1, T+1):
        AYt = 0
        for i in range(1, p+1):
            AYt += a[i] * Y[t-i]
        likelihood_part += (Y[t] - a[0] - AYt)**2
        tmp += 1
        if tmp >= 20:
            tmp = 0
            likelihood_part = sympy.simplify(likelihood_part)
    likelihood_part = sympy.simplify(likelihood_part)
    likelihood_part *= 1/(2*sigma_square)
    sum_As = 0
    for i in range(0, p+1):
        sum_As += a[i]**2
    sum_As *= 1/(2*sigma_square)
    to_minimum = ((T+1)/2)*sympy.log(sigma_square) + likelihood_part + sum_As + ((T-2*p)/2+1)*sympy.log(sigma_square) + ((T-2*p)/2)/(0.37*sigma_square)
    to_minimum = sympy.simplify(to_minimum)
    params = []
    for single_a in a:
        params.append(single_a)
    formulas = []
    for single_a in a:
        formulas.append(sympy.simplify(sympy.diff(to_minimum, single_a) * sigma_square))
    print(params)
    for formula in formulas:
        print(formula)
    results = sympy.solve(formulas, params)
    print('RESULT FOR AR(%s):' % p, results)
#    subs = dict()
#    for i in range(0, p+1):
#        key = 'a%s' % i
#        print(key, results[i])
#        subs.setdefault(str(key), results[i])
#    print(subs)
    diff_sigmasquare = sympy.simplify(sympy.diff(to_minimum, sigma_square))
    print(diff_sigmasquare)
    sigma_square_result = sympy.solve(diff_sigmasquare, sigma_square)
    print(sigma_square_result)
    sigma_square_result = sigma_square_result[0].evalf(subs=results)
    print('sigma_square_result AR(%s):' % p, sigma_square_result)
    result_file_path = './Results/res_invGamma_prior.txt'
    with open(result_file_path, 'a+') as f:
        f.write('RESULT FOR AR(%s):' % p + '\n')
        f.write('SIGMA_SQUARE FOR AR(%s):' % p + str(sigma_square_result) + '\n')
        f.write('PARAMETERS(AR(%s)):' % p + str(params) + '\n')
        f.write('RESULTS(AR(%s)):' % p + str(results) + '\n')
        f.close()


def main():
    dates = [0]
    prices = [0]
    with open('./Datas/IBM.txt', 'r') as f:
        next(f)
        for line in f:
            data_list = re.split(',', line)
            data_list[0] = datetime.datetime.strptime(data_list[0], '%Y-%m-%d')
            data_list[1] = float(data_list[1][:-1])
            dates.append(data_list[0])
            prices.append(data_list[1])
    dates = np.array(dates)
    prices = np.array(prices)
    T = len(prices) - 1
    process_list = []
    for i in range(1, 20):
        p = Process(target=calculate_ARp, args=(i, prices.copy(), T))
        process_list.append(p)
    for p in process_list:
        p.start()
    for p in process_list:
        p.join()


if __name__ == '__main__':
    main()
import numpy as np


# https://stackoverflow.com/questions/42633083/how-to-perform-a-levenes-test-using-scipy
# does not work need to fix
# https://www.quantopian.com/posts/box-m-test-to-compare-covariance-matrices-across-time
# def boxMtest(covs, n0, n1):
#     (p, p) = np.shape(covs[0])
#     k = len(covs)
#     N = n0 + n1
#     S = (n0-1.) * covs[0] + (n1 - 1.) * covs[1]
#     S = S / (N - k)
#     M1 = (N - k) * np.log(np.linalg.det(S))
#     M2 = (n0 - 1.) * np.log(np.linalg.det(covs[0])) + (n1 - 1.) * np.log(np.linalg.det(covs[1]))
#     M = M1 - M2
#     A1 = (2. * (p**2) + 3. * p + 1.) / (6. * (p+1)* (k-1))
#     A1 *= (1. / (n0 - 1.) + 1. / (n1 - 1.)) - 1. / (N - k)
#     v1 = p * (p + 1.) * (k -1.) / 2.
#     A2 = (p - 1.) * (p + 2.) / (6. * (k-1.))
#     A2 *= (1. / (n0 - 1.)**2 + 1. / (n1 - 1.))**2 - 1. / (N - k)**2
#     Adiff = A2 - A1**2
#
#     if Adiff > 0:
#         v2 = (v1 + 2) / Adiff
#         b = v1 / (1-A1 - (v1/v2))
#         F = M / b
#     else:
#         v2 = (v1 + 2) / -Adiff
#         b = v2 / (1-A1 + (2./v2))
#         F = (v2 * M) / (v1 * (b - M))
#
#     p_value = 1 - scipy.stats.f.cdf(F, v1, v2)
#     print p_value
#     if p_value <= 0.05:
#         print "covariance are unequal"
#     else:
#         print "covariance equal"```

import pylab as plt
import numpy as np


def boltzman(x, xmid, tau):
    return 1. / (1. + np.exp(-(x-xmid)/tau))

n=9
tau = 5
xmid = 0
x = np.arange(0, n, 1)

result = boltzman(x, xmid=xmid, tau=tau)
print(np.round((result)-0.5, 2))

c = np.arange(0, 10, 1)
result = 1 - boltzman(c, xmid=xmid, tau=tau)
print(np.round((result)+0.5, 2))

x = np.arange(-50, 50, 1)
xmid = 10
x = np.arange(-500, 500, 1)
S = boltzman(x, xmid, 100)
Z = 1-boltzman(x, xmid, 100)
# Z = boltzman(x, xmid+30, 1)

plt.plot(x, S, x, Z, color='red', lw=2)
plt.show()
#


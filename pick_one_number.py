# -*- coding: utf-8 -*-

import random
import matplotlib.pyplot as plt

result = []
sum5num = 0
for _ in range(0, 5000):
    for i in range(0,5):
        sum5num += random.randint(1, 7)
    result.append((sum5num%15) + 1)

plt.hist(result, 100)

plt.xlabel('number')
plt.xlim(0.0,16)
plt.ylabel('Frequency')
plt.title('pick one number')
plt.show()
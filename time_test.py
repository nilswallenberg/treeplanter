import timeit
import numpy as np

test_code1 = """
import numpy as np
a = np.random.rand(250,250)
b = np.multiply(a,a)
"""

test_code2 = """
import numpy as np
import numexpr as ne
a = np.random.rand(250,250)
b = ne.evaluate('a * a')
"""

test_code3 = """
import numpy as np
a = np.random.rand(250,250)
b = a * a
"""

test1 = timeit.timeit(test_code1, number=10000)
test2 = timeit.timeit(test_code2, number=10000)
test3 = timeit.timeit(test_code3, number=10000)

print(test1)
print(test2)
print(test3)
from math import floor

# Triangular number -- how many comparison on single thread given 'x' elements
def Tr(x): 
    n = x // 2
    # return n * 0.5 * (n + 1)
    if n <= 1:
        return 1
    return n + Tr(n)

# number count
n = 256

# procecssor count
p = 1

# prints how many steps take parallel prefix scan versus sequential
print (Tr(16))
print (int(2 * (Tr(n/p) + Tr(p))), "vs", n)
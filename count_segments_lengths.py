'''
In [1]: a = np.array([1,1,2,2,2,2,3,1,1,1])

In [2]: np.diff(np.flatnonzero(np.diff(a))+1, prepend=0, append=a.shape[0])
Out[3]: array([2, 4, 1, 3], dtype=int64)
'''

import numpy as np

testarr = np.asarray([(0,1),
                      (2,3),
                      (4,5)])

print testarr

testarr[1,:] = (6,7)

print testarr
print testarr.shape
print np.zeros([3,2])
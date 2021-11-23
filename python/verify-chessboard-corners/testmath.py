import numpy as np

# Development code that shows the equivalent computation
# of support using matrix multiplication

# Define number of objects (nObj) and number of labels (l)
nObj = 4
nLab = 3

# Define compat
compat = np.ones((nObj,nLab,nObj,nLab,nObj,nLab))
val = 0
for i in range(0,nObj):
    for j in range(0,nLab):
        for k in range(0,nObj):
            for l in range(0,nLab):
                for m in range(0,nObj):
                    for n in range(0,nLab):
                        val += 1
                        compat[i,j,k,l,m,n] = val

# Define strength
strength = np.zeros((nObj,nLab))
val = 0
for i in range(0,nObj):
    for j in range(0,nLab):
        val += 1*.234
        strength[i,j] = val

# Compute the support by embedded for loops
support = np.zeros((nObj,nLab))
for i in range(0,nObj):
    for j in range(0,nLab):
        for k in range(0,nObj):
            for l in range(0,nLab):
                for m in range(0,nObj):
                    for n in range(0,nLab):
                        support[i,j] += strength[k,l]*strength[m,n]*compat[i,j,k,l,m,n]

print('strength')
print(strength)
print('support')
print(support)


# Compute the support by matrix multiplications
expanded_strength = np.zeros((nObj, nLab, nObj, nLab))
for i in range(0,nObj):
    for j in range(0,nLab):
            expanded_strength[i,j,:,:] = strength[i,j]

support = np.zeros((nObj,nLab))
# For 2 pairs case, z = np.multiply(strength, compat)
z = np.multiply(expanded_strength, compat)
z = np.multiply(strength, z)
print('shape of z',z.shape)
z = np.reshape(z, (nObj,nLab,-1))
print('shape of (reshaped) z',z.shape)
z = z.sum(axis=2)
print('Sum along axis 2')
print(z)

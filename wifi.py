import numpy as np
import matplotlib.pyplot as plt
import json


up = 30
down = -30
code = np.array([+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1], dtype='int8')  # 11
code = np.repeat(code, 5)

with open('wifi/Кущ.dat') as f:
    data = np.array(f.readlines(), dtype='float64')

# Returns the discrete, linear convolution of two one-dimensional sequences.
# Mode ‘same’ returns output of length max(M, N). Boundary effects are still visible.
cnl = np.convolve(data, code[::-1], mode='same')  # or 'full'

plt.plot(cnl)
# plt.plot(np.convolve(data, code[::-1], mode='full'))
plt.title('КРАСИВО')
plt.show()

bit = []

for i in range(cnl.shape[0]):
    if cnl[i] > up and cnl[i - 1] < cnl[i] and cnl[i + 1] < cnl[i]:
        bit.append(1)
    elif cnl[i] < down and cnl[i - 1] > cnl[i] and cnl[i + 1] > cnl[i]:
        bit.append(0)

# Packs the elements of a binary-valued array into bits in a uint8 array.
# Then unpack. Construct Python bytes containing the raw data bytes in the array.
# Finally decode to ASCII
message = np.packbits(np.array(bit)).tobytes().decode('ascii')
print(message)

with open('wifi.json', 'w') as file:
    json.dump({"message": message}, file)

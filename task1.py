import math
eps = 1
while 1 + eps / 2 > 1:
    eps /= 2

def pf(msg, val):
    print(msg, val, float.hex(val))

pf('eps =', eps)
for i in range(1, 10):
    pf(f'{i}/2', 1 + eps * (i // 2) + (eps / 2 if i % 2 else 0))

while eps / 2 > 0:
    eps /= 2
print('min =', float.hex(eps), eps)

while eps * 2 < math.inf:
    eps *= 2
print('max =', float.hex(eps), eps)


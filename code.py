import numpy as np

# Importando os dados do csv
ySign = np.loadtxt("rx_sgn_y.csv", dtype=complex)

# Especificacao dos parametros dados no projeto
N = 21
Ns = 100
fc = 1000
a = 0.5
t = 1
vx = 2
vy = 2 

print(ySign[1])
print("fim")
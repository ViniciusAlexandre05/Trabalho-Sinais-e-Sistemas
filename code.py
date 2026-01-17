import numpy as np
import matplotlib.pyplot as plt

# Importando os dados do csv
ySignal = np.loadtxt("rx_sgn_y.csv", dtype=complex)

# Total de amostras
total_samples = len(ySignal)

# Especificacao dos parametros dados no projeto
N = 21
Ns = 100
fc = 1000
Ts = 1/ (N*fc)


# Criando vetor de tempo discreto
n = np.arange(total_samples)

# Etapa 1 - Decomposicao do sinal ------------
arg = (2 * np.pi / N) * n

yReal = np.sqrt(2) * np.cos(arg) * ySignal

yImg = -np.sqrt(2) * np.sin(arg) * ySignal



# Etapa 2 - Filtro(Media Movel) --------------
h = np.ones(N) / N

# Convolucao
yReal_mm = np.convolve(yReal, h, mode='full')

yImg_mm = np.convolve(yImg, h, mode='full')



# Amostragem ---------------------------------
simbolos_estimados = []

for m in range(Ns):
    idx = (m * N) + (N - 1)

    val_real = yReal_mm[idx]
    val_img = yImg_mm[idx]

    simbolos_estimados.append(val_real + 1j * val_img)

simbolos_estimados = np.array(simbolos_estimados)


# Etapa 3 - Decisor ------------------------------

# Constelacao Ideal
constelacao = [
    1/np.sqrt(2) + 1j/np.sqrt(2),  #S1
    -1/np.sqrt(2) + 1j/np.sqrt(2), #S2
    -1/np.sqrt(2) - 1j/np.sqrt(2), #S3
    1/np.sqrt(2) - 1j/np.sqrt(2)   #S4
]

simbolos_decodificados = []

for s_estimado in simbolos_estimados:
    distancias  = [abs(s_estimado - ponto) for ponto in constelacao]
    indice_menor = np.argmin(distancias)
    simbolos_decodificados.append(constelacao[indice_menor])


# Teste da rota --------------------------------------
cx, cy = 0, 0 # Posicoes Iniciais
vx, vy = 2, 2 # Velocidades 
path_x = [cx]
path_y = [cy]
alpha = 0.5
tau = 1

for sym in simbolos_decodificados:
    vx = vx + alpha * sym.real
    vy = vy + alpha * sym.imag

    cx = cx + tau * vx
    cy = cy + tau * vy

    path_x.append(cx)
    path_y.append(cy)

print(f"Coordenada final: ({cx:.4f}, {cy:.4f})")

# Gerar graficos ---------------------------------

# Constelacoes (Estimados x Esperados)

plt.figure(figsize=(8, 8))

# Plotar os símbolos estimados
plt.scatter(simbolos_estimados.real, simbolos_estimados.imag, 
            color='blue', alpha=0.5, label='Símbolos Estimados', marker='.')

# Plotar a constelação ideal
const_x = [p.real for p in constelacao]
const_y = [p.imag for p in constelacao]
plt.scatter(const_x, const_y, color='red', marker='x', s=100, linewidths=2, label='Constelação Ideal')

# Adicionar legendas para cada símbolo ideal
for i, ponto in enumerate(constelacao):
    plt.annotate(f'S{i+1}', (ponto.real, ponto.imag), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontweight='bold', color='red')

# Formatação do gráfico
plt.axhline(0, color='black', linestyle='--', linewidth=0.5) # Eixo X
plt.axvline(0, color='black', linestyle='--', linewidth=0.5) # Eixo Y
plt.grid(True, linestyle=':', alpha=0.6)
plt.title('Diagrama de Constelação: Símbolos Recebidos vs. Ideais', fontsize=14)
plt.xlabel('Componente com parte real dos símbolos')
plt.ylabel('Componente com parte imaginaria dos símbolos')
plt.legend(loc='upper right')
plt.axis('equal')

plt.show()

# --- Gráfico da Trajetória do Drone ---

plt.figure(figsize=(8, 6))
plt.plot(path_x, path_y, marker='o', linestyle='-', color='orange', markersize=4)
plt.title('Trajetória Estimada do Drone no Plano XY', fontsize=14)
plt.xlabel('Posição X')
plt.ylabel('Posição Y')
plt.grid(True)
plt.show()

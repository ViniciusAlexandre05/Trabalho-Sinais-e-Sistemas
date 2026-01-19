import numpy as np
import matplotlib.pyplot as plt

# Importando os dados do csv
# Certifique-se de que o arquivo existe ou substitua por dados simulados para teste
try:
    ySignal = np.loadtxt("rx_sgn_y.csv", dtype=complex)
except OSError:
    # Gerando dados aleatórios apenas para o código rodar caso você não tenha o CSV aqui
    print("Aviso: CSV não encontrado. Gerando dados aleatórios para teste.")
    ySignal = (np.random.randn(2100) + 1j * np.random.randn(2100))

# Especificacao dos parametros dados no projeto
N = 21       # Amostras por símbolo
Ns = 100     # Número de símbolos
fc = 1000
Ts = 1/ (N*fc)

# Validação de segurança: garantir que temos amostras suficientes
total_needed = N * Ns
if len(ySignal) < total_needed:
    raise ValueError(f"O sinal possui {len(ySignal)} amostras, mas são necessárias {total_needed}.")

# Cortamos o sinal para garantir que é múltiplo exato de N e Ns
ySignal = ySignal[:total_needed]

# Vetor de tempo discreto
n = np.arange(len(ySignal))

# =============================================================================
# Etapa 1 - Decomposicao do sinal
# =============================================================================
arg = (2 * np.pi / N) * n
yReal = np.sqrt(2) * np.cos(arg) * ySignal
yImg = -np.sqrt(2) * np.sin(arg) * ySignal

# =============================================================================
# Etapa 2 e Amostragem - Filtro de Média Móvel "Resetável" (Sem ISI)
# =============================================================================

# Em vez de convolução contínua, transformamos o vetor 1D em uma matriz 2D.
# Dimensões: (Número de Símbolos, Amostras por Símbolo) = (100, 21)
# Cada linha representa um símbolo isolado.
yReal_matrix = yReal.reshape(Ns, N)
yImg_matrix  = yImg.reshape(Ns, N)

# O Filtro de Média Móvel soma as N amostras e divide por N.
# Como queremos o valor amostrado no final do período de símbolo,
# basta calcular a média de cada linha.
# Isso garante ZERO interferência entre o símbolo anterior e o atual.

val_real_amostrados = np.mean(yReal_matrix, axis=1)
val_img_amostrados  = np.mean(yImg_matrix, axis=1)

# Reconstrói os símbolos complexos
simbolos_estimados = val_real_amostrados + 1j * val_img_amostrados

# Nota: Se você quisesse ver a forma de onda triangular se formando (o "ramp up"),
# usaria np.cumsum(matrix, axis=1)/N. Mas para o decisor, só a média basta.

# =============================================================================
# Etapa 3 - Decisor
# =============================================================================

# Constelacao Ideal
constelacao = [
    1/np.sqrt(2) + 1j/np.sqrt(2),  # S1
   -1/np.sqrt(2) + 1j/np.sqrt(2),  # S2
   -1/np.sqrt(2) - 1j/np.sqrt(2),  # S3
    1/np.sqrt(2) - 1j/np.sqrt(2)   # S4
]

simbolos_decodificados = []

for s_estimado in simbolos_estimados:
    distancias  = [abs(s_estimado - ponto) for ponto in constelacao]
    indice_menor = np.argmin(distancias)
    simbolos_decodificados.append(constelacao[indice_menor])

# =============================================================================
# Teste da rota
# =============================================================================
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

# =============================================================================
# Gerar graficos
# =============================================================================

# --- Constelação ---
plt.figure(figsize=(8, 8))
plt.scatter(simbolos_estimados.real, simbolos_estimados.imag, 
            color='blue', alpha=0.5, label='Símbolos Estimados', marker='.')

const_x = [p.real for p in constelacao]
const_y = [p.imag for p in constelacao]
plt.scatter(const_x, const_y, color='red', marker='x', s=100, linewidths=2, label='Constelação Ideal')

for i, ponto in enumerate(constelacao):
    plt.annotate(f'S{i+1}', (ponto.real, ponto.imag), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontweight='bold', color='red')

plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
plt.grid(True, linestyle=':', alpha=0.6)
plt.title('Constelação (Sem Interferência entre Símbolos)')
plt.xlabel('Real')
plt.ylabel('Imaginário')
plt.legend(loc='upper right')
plt.axis('equal')
plt.show()

# --- Trajetória ---
plt.figure(figsize=(8, 8))
plt.plot(path_x, path_y, linestyle='-', color='orange', alpha=0.7, label='Caminho do Drone', zorder=1)
plt.scatter(path_x, path_y, color='orange', s=10, alpha=0.5)
plt.scatter(path_x[0], path_y[0], color='green', s=120, edgecolors='black', zorder=5, label='Início')
plt.scatter(path_x[-1], path_y[-1], color='red', s=120, edgecolors='black', zorder=5, label='Fim')

plt.axis('equal') 
margin = 5
plt.xlim(min(path_x) - margin, max(path_x) + margin)
plt.ylim(min(path_y) - margin, max(path_y) + margin)

plt.title('Trajetória Estimada do Drone')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()
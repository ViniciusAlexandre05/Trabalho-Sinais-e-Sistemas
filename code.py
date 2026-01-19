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
simbolos_estimados = []    # Com filtro (Sinal limpo)
simbolos_sem_filtro = []   # Sem filtro (Sinal sujo saindo do mixer)

for m in range(Ns):
    # Indice de amostragem (final do simbolo)
    idx = (m * N) + (N - 1) 

    # 1. Captura do sinal FILTRADO (O que voce ja tinha)
    val_real = yReal_mm[idx]
    val_img = yImg_mm[idx]
    simbolos_estimados.append(val_real + 1j * val_img)

    # 2. Captura do sinal SEM FILTRO (Direto do Mixer)
    # Isso vai mostrar a interferencia de 2*fc
    raw_real = yReal[idx]
    raw_img = yImg[idx]
    simbolos_sem_filtro.append(raw_real + 1j * raw_img)

simbolos_estimados = np.array(simbolos_estimados)
simbolos_sem_filtro = np.array(simbolos_sem_filtro)


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

# Grafico sem comparacoes
plt.figure(figsize=(8, 8))

# Plotar os simbolos estimados
plt.scatter(simbolos_estimados.real, simbolos_estimados.imag, 
            color='blue', alpha=0.5, label='Simbolos Estimados', marker='.')

# Plotar a constelacao ideal
const_x = [p.real for p in constelacao]
const_y = [p.imag for p in constelacao]
plt.scatter(const_x, const_y, color='red', marker='x', s=100, linewidths=2, label='Constelacao Ideal')

# Adicionar legendas para cada simbolo ideal
for i, ponto in enumerate(constelacao):
    plt.annotate(f'S{i+1}', (ponto.real, ponto.imag), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontweight='bold', color='red')

# Formatacao do grafico
plt.axhline(0, color='black', linestyle='--', linewidth=0.5) # Eixo X
plt.axvline(0, color='black', linestyle='--', linewidth=0.5) # Eixo Y
plt.grid(True, linestyle=':', alpha=0.6)
plt.title('Diagrama de Constelacao: Simbolos Recebidos vs. Ideais', fontsize=14)
plt.xlabel('Componente com parte real dos simbolos')
plt.ylabel('Componente com parte imaginaria dos simbolos')
plt.legend(loc='upper right')
plt.axis('equal')

plt.show()

# Constelacoes com comparacao
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- GRAFICO 1: SEM FILTRO (Esquerda) ---
ax1.scatter(simbolos_sem_filtro.real, simbolos_sem_filtro.imag, 
            color='orange', alpha=0.6, label='Pos-Mixer (Sem Filtro)', marker='.')

# Desenha a constelacao ideal em cima para referencia
const_x = [p.real for p in constelacao]
const_y = [p.imag for p in constelacao]
ax1.scatter(const_x, const_y, color='red', marker='x', s=100, linewidths=2, label='Ideal')

ax1.set_title('Antes do Filtro (Saida do Misturador)', fontsize=14)
ax1.set_xlabel('Real (I)')
ax1.set_ylabel('Imaginario (Q)')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.axhline(0, color='black', linewidth=0.5)
ax1.axvline(0, color='black', linewidth=0.5)
ax1.legend()
ax1.axis('equal')

# --- GRAFICO 2: COM FILTRO (Direita) ---
ax2.scatter(simbolos_estimados.real, simbolos_estimados.imag, 
            color='blue', alpha=0.6, label='Pos-Filtro (Media Movel)', marker='.')

# Desenha a constelacao ideal
ax2.scatter(const_x, const_y, color='red', marker='x', s=100, linewidths=2, label='Ideal')

# Adiciona legendas S1, S2... apenas no grafico limpo para nao poluir
for i, ponto in enumerate(constelacao):
    ax2.annotate(f'S{i+1}', (ponto.real, ponto.imag), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontweight='bold', color='red')

ax2.set_title('Depois do Filtro (Media Movel N=21)', fontsize=14)
ax2.set_xlabel('Real (I)')
ax2.set_ylabel('Imaginario (Q)')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.axvline(0, color='black', linewidth=0.5)
ax2.legend()
ax2.axis('equal')

plt.tight_layout() # Ajusta o espacamento para nao sobrepor textos
plt.show()

# --- Grafico da Trajetoria do Drone ---
plt.figure(figsize=(8, 8))

# 1. Plotar a linha da trajetoria
plt.plot(path_x, path_y, linestyle='-', color='orange', alpha=0.7, label='Caminho do Drone', zorder=1)
plt.scatter(path_x, path_y, color='orange', s=10, alpha=0.5) # Pontos intermediarios pequenos

# 2. Ponto de Inicio
plt.scatter(path_x[0], path_y[0], color='green', s=120, edgecolors='black', zorder=5, label=f'Inicio: ({path_x[0]:.4f}, {path_y[0]:.4f})')

# 3. Ponto de Fim
plt.scatter(path_x[-1], path_y[-1], color='red', s=120, edgecolors='black', zorder=5, label=f'Fim: ({path_x[-1]:.4f}, {path_y[-1]:.4f})')

# 4. Anotacoes de texto
plt.text(path_x[0], path_y[0] + 1, 'Inicio', color='green', fontweight='bold', ha='center')
plt.text(path_x[-1], path_y[-1] + 1, 'Fim', color='red', fontweight='bold', ha='center')

# --- Ajustes de Escala e Enquadramento ---

# 'equal' garante que 1 metro em X seja igual a 1 metro em Y na tela
plt.axis('equal') 

# Adiciona uma margem de seguranca para os rotulos nao ficarem cortados
margin = 5
plt.xlim(min(path_x) - margin, max(path_x) + margin)
plt.ylim(min(path_y) - margin, max(path_y) + margin)

plt.title('Trajetoria Estimada do Drone', fontsize=14)
plt.xlabel('Posicao X')
plt.ylabel('Posicao Y')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best')

plt.show()
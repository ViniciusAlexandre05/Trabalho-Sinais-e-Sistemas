import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, fftfreq

# CONFIGURAÇÕES GERAIS
N = 21              # Tamanho do filtro
fc = 1000           # Frequência da portadora
fs = N * fc         # 21000 Hz

# GRÁFICO 1: MAGNITUDE DA RESPOSTA EM FREQUÊNCIA TEÓRICA:
# Definição do filtro
h = np.ones(N) / N 

# Calculo da resposta em frequênci
num_pontos_fft = 4096 # alto numero de pontos para suavizar
H = fft(h, num_pontos_fft)

# Eixo da frequência normalizada 
H_mag = np.abs(H[:num_pontos_fft//2])
w_norm = np.linspace(0, 1, len(H_mag)) 

plt.figure(figsize=(10, 5))
plt.plot(w_norm, H_mag, 'royalblue', linewidth=2)

# Estilização para ficar igual ao seu relatório
plt.title(f'Resposta em Frequência do Filtro (N={N})')
plt.xlabel(r'Frequência Normalizada ($\times \pi$ rad/amostra)')
plt.ylabel('Magnitude')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axvline(2/N, color='red', linestyle='--', label='Corte aprox.')
plt.legend()
plt.tight_layout()
plt.show()


# GRÁFICO 2: ESPECTRO DE FREQUÊNCIA

# Gerar um sinal sintético similar ao do drone para o gráfico
np.random.seed(42)
num_simbolos = 200
t = np.arange(num_simbolos * N)


# Sinal de dados
dados = np.repeat(np.random.choice([-1, 1], num_simbolos), N)

# Misturador: Sinal * cos(2*pi*fc*t) * cos(2*pi*fc*t) gera DC + 2fc
# Aqui simulamos direto o resultado do mixer: Parte DC + Parte em 2*fc + Ruído
sinal_pos_mixer = (dados * 1.0) + (dados * np.cos(2 * np.pi * (2*fc/fs) * t)) + 0.5 * np.random.randn(len(t))

# Aplicar o Filtro
sinal_filtrado = np.convolve(sinal_pos_mixer, h, mode='same')

# Calculo do FFT para o Gráfico
freqs = fftshift(fftfreq(len(t), d=1/fs))
X_mixer = fftshift(fft(sinal_pos_mixer))
X_filt = fftshift(fft(sinal_filtrado))

# Magnitude normalizada para escala do gráfico
mag_mixer = np.abs(X_mixer)
mag_filt = np.abs(X_filt)

# Plotagem
plt.figure(figsize=(10, 6))

# Plot do sinal ruidoso
plt.plot(freqs, mag_mixer, color='orange', alpha=0.6, linewidth=1, label='Pós-Mixer (Ruidoso)')

# Plot do sinal filtrado
plt.plot(freqs, mag_filt, color='green', linewidth=1.5, label='Filtrado (Limpo)')

# Configurações de Eixo e Texto
plt.title('Espectro de Frequência')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.legend(loc='upper right', frameon=True, fancybox=True, framealpha=1, borderpad=1)
plt.grid(True, alpha=0.3)

# Limites iguais à imagem ]
plt.xlim(-3000, 3000)
plt.tight_layout()
plt.show()
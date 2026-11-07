import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal.windows import boxcar, hann, hamming, blackman

# 1. Parametry sygnału (Wariant 11)
f1 = 200.0
f2 = 200.25
f3 = 199.75
A1 = 4.0
A2 = 0.1 * A1
A3 = 0.01 * A1
fs = 400.0
N = 2000

# 2. Generowanie sygnału
k = np.arange(N)
x = (A1 * np.sin(2 * np.pi * f1 / fs * k) + 
     A2 * np.sin(2 * np.pi * f2 / fs * k) + 
     A3 * np.sin(2 * np.pi * f3 / fs * k))

# 3. Generowanie okien (okresowe: sym=False)
w_rect = boxcar(N)
w_hann = hann(N, sym=False)
w_hamm = hamming(N, sym=False)
w_black = blackman(N, sym=False)

# 4. Funkcja do normalizacji widma DFT (według instrukcji)
def fft2db(X):
    N_size = X.size
    Xtmp = 2 / N_size * X
    Xtmp[0] *= 1/2
    if N_size % 2 == 0:
        Xtmp[N_size//2] = Xtmp[N_size//2] / 2
    # Zabezpieczenie przed log(0)
    magnitude = np.abs(Xtmp)
    magnitude[magnitude < 1e-12] = 1e-12
    return 20 * np.log10(magnitude)

# Obliczenie widm DFT
X_rect = fft(x * w_rect)
X_hann = fft(x * w_hann)
X_hamm = fft(x * w_hamm)
X_black = fft(x * w_black)

# Wektor częstotliwości
df = fs / N
f = np.arange(N) * df

# Rysowanie widm DFT (Zadanie 1)
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(f, fft2db(X_rect), 'C0o-', ms=3, label='Rectangular')
plt.xlim(198, 202)
plt.ylim(-80, 20)
plt.ylabel('A / dB')
plt.legend()
plt.grid(True)
plt.title('Znormalizowane widma DFT dla różnych okien (Zbliżenie na 198-202 Hz)')

plt.subplot(4, 1, 2)
plt.plot(f, fft2db(X_hann), 'C1o-', ms=3, label='Hann')
plt.xlim(198, 202)
plt.ylim(-80, 20)
plt.ylabel('A / dB')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(f, fft2db(X_hamm), 'C2o-', ms=3, label='Hamming')
plt.xlim(198, 202)
plt.ylim(-80, 20)
plt.ylabel('A / dB')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(f, fft2db(X_black), 'C3o-', ms=3, label='Blackman')
plt.xlim(198, 202)
plt.ylim(-80, 20)
plt.xlabel('f / Hz')
plt.ylabel('A / dB')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 5. Funkcja do analizy DTFT okien z zero-paddingiem (według instrukcji)
def winDTFTdB(w):
    N_w = len(w)
    Nz = 100 * N_w
    W = np.zeros(Nz)
    W[0:N_w] = w
    W_fft = np.abs(fftshift(fft(W)))
    W_norm = W_fft / np.max(W_fft)
    W_norm[W_norm < 1e-12] = 1e-12
    W_dB = 20 * np.log10(W_norm)
    Omega = 2 * np.pi / Nz * np.arange(Nz) - np.pi
    return Omega, W_dB

O_rect, W_rect_dtft = winDTFTdB(w_rect)
O_hann, W_hann_dtft = winDTFTdB(w_hann)
O_hamm, W_hamm_dtft = winDTFTdB(w_hamm)
O_black, W_black_dtft = winDTFTdB(w_black)

# Rysowanie widm DTFT okien (Zadanie 2)
plt.figure(figsize=(10, 6))
plt.plot(O_rect, W_rect_dtft, label='Rectangular')
plt.plot(O_hann, W_hann_dtft, label='Hann')
plt.plot(O_hamm, W_hamm_dtft, label='Hamming')
plt.plot(O_black, W_black_dtft, label='Blackman')

plt.xlim(-0.02, 0.02)
plt.ylim(-120, 10)
plt.xlabel(r'$\Omega$')
plt.ylabel(r'$|W(\Omega)|$ / dB')
plt.title('Widma DTFT okien znormalizowane do maksimum listka głównego')
plt.legend()
plt.grid(True)
plt.show()
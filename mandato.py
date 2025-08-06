import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

N = 10000
I0 = 10
R0 = 0
S0 = N - I0 - R0
beta = 0.35
gamma = 0.1
dias = 160
t = np.linspace(0, dias, dias)

def SIR_refuerzo_continuo(y, t_actual, beta, gamma):
    S, I, R = y
    Rt = (beta / gamma) * (S / N)

    if Rt >= 1:
        vacunados = 0.5 * S
        S -= vacunados
        R += vacunados

    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    
    return [dSdt, dIdt, dRdt]

def simular():
    y0 = [S0, I0, R0]
    resultado = odeint(SIR_refuerzo_continuo, y0, t, args=(beta, gamma))
    return resultado

def calcular_Rt(S):
    return (beta / gamma) * (S / N)

def graficar_resultado(t, resultado, titulo):
    S, I, R = resultado.T
    Rt = calcular_Rt(S)

    plt.figure(figsize=(10,6))
    plt.plot(t, I, 'r', label='Infectados')
    plt.plot(t, S, 'b', label='Susceptibles')
    plt.plot(t, R, 'g', label='Recuperados/Vacunados')

    plt.xlabel('Días')
    plt.ylabel('Personas / $R_t$')
    plt.title(titulo)
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.show()

# Ejecutar
resultado = simular()
graficar_resultado(t, resultado, 'SIR con vacunación cuando $R_t \geq 1$')

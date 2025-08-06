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

vacuna_aplicada = False
t_intervencion = None

def SIR_Mandato(y, t, beta, gamma):
    global vacuna_aplicada, t_intervencion
    S, I, R = y
    Rt = (beta / gamma) * (S / N)

    if Rt > 1 and not vacuna_aplicada:
        vacunados = 0.5 * S
        S -= vacunados
        R += vacunados
        vacuna_aplicada = True
        t_intervencion = t

    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def simular_con_mandato():
    global vacuna_aplicada, t_intervencion
    vacuna_aplicada = False
    t_intervencion = None
    y0 = [S0, I0, R0]
    resultado = odeint(SIR_Mandato, y0, t, args=(beta, gamma))
    return resultado, t_intervencion

def calcular_Rt(S):
    return (beta / gamma) * (S / N)

def graficar_resultado(t, resultado, titulo, t_intervencion=None):
    S, I, R = resultado.T
    Rt = calcular_Rt(S)

    plt.figure(figsize=(10,6))
    plt.plot(t, I, 'r', label='Infectados')
    plt.plot(t, S, 'b', label='Susceptibles')
    plt.plot(t, R, 'g', label='Recuperados/Vacunados')

    if t_intervencion:
        plt.axvline(x=t_intervencion, color='gray', linestyle=':', label='Intervención')

    # Encontrar cruce de Rt por 1 (buscando primer índice donde Rt <= 1)
    idx_rt1 = np.where(Rt <= 1)[0]
    if idx_rt1.size > 0:
        plt.axvline(x=t[idx_rt1[0]], color='purple', linestyle='--', label='$R_t = 1$')

    plt.xlabel('Días')
    plt.ylabel('Personas')
    plt.title(titulo)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Ejecutar simulación y graficar
resultado, t_interv = simular_con_mandato()
graficar_resultado(t, resultado, 'SIR con Mandato (Vacunación 50% cuando $R_t > 1$)', t_interv)

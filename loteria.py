import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Variable global para controlar cuándo se supera el umbral
start_vaccination_day = None

# Modelo SIR con vacunación dinámica (ν aumenta cada día después de I >= 100)
def sir_vac_model(y, t, beta, gamma):
    global start_vaccination_day
    S, I, R = y
    N = S + I + R

    # Detectar el día en que se superan los 100 infectados
    if I >= 100 and start_vaccination_day is None:
        start_vaccination_day = t  # guardamos el tiempo en que inicia vacunación
        print(start_vaccination_day)

    # Calcular tasa de vacunación ν(t)
    if start_vaccination_day is not None:
        days_since = t - start_vaccination_day
        nu = 0.02 * days_since
    else:
        nu = 0

    # Ecuaciones diferenciales
    dSdt = -beta * S * I / N - nu * S
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I + nu * S
    return [dSdt, dIdt, dRdt]

# Parámetros
beta = 0.35
gamma = 0.1

# Condiciones iniciales
N = 10000
I0 = 10
R0 = 0
S0 = N - I0 - R0
y0 = [S0, I0, R0]

# Tiempo
t = np.linspace(0, 160, 160)

# Reiniciar variable global
start_vaccination_day = None

# Resolver ODE
sol = odeint(sir_vac_model, y0, t, args=(beta, gamma))
S, I, R = sol.T

# Graficar
plt.figure(figsize=(10,6))
plt.plot(t, S, label='Susceptibles')
plt.plot(t, I, label='Infectados')
plt.plot(t, R, label='Recuperados + Vacunados')

# Punto donde inicia vacunación
plt.scatter(start_vaccination_day, I[int(start_vaccination_day)], color='red', label='Inicio vacunación (I=100)')

plt.title('Modelo SIR con vacunación dinámica')
plt.xlabel('Días')
plt.ylabel('Personas')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parámetros del modelo
N = 10000
I0 = 10
R0 = 0
S0 = N - I0 - R0
beta = 0.35
gamma = 0.1
dias = 160
t = np.linspace(0, dias, dias + 1)

def calcular_Rt(S):
    rt = (beta / gamma) * (S / N)
    print(rt)
    return rt

def SIR_linea_base(y, t, beta, gamma):
    """
    Modelo SIR básico sin intervenciones (línea base).
    """
    S, I, R = y
    
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    
    return [dSdt, dIdt, dRdt]

def encontrar_dia_vacunacion():
    """
    Función para calcular el día de refuerzo.
    """
    y0 = [S0, I0, R0]
    t_eval = np.linspace(0, dias, dias + 1)
    resultado = odeint(SIR_linea_base, y0, t_eval, args=(beta, gamma))
    
    for i, (S, I, R) in enumerate(resultado):
        Rt = calcular_Rt(S)
        if Rt > 1:
            return i, resultado[i]
    return None, resultado[-1]

def simular_con_vacunacion():
    """
    Simulación vacunando el primer día donde Rt > 1.
    """
    
    # Buscar el primer día donde Rt > 1 y el estado en ese momento
    dia_vac, estado_vac = encontrar_dia_vacunacion()

    # Si nunca se supera Rt > 1
    if dia_vac is None:
        print("Rₜ nunca superó 1. No se aplicó vacunación.")
        return odeint(SIR_linea_base, [S0, I0, R0], t, args=(beta, gamma)), None

    # 1. Simulación desde el día 0 hasta el día en que se aplica el refuerzo
    t1 = np.linspace(0, dia_vac, dia_vac + 1)
    y0 = [S0, I0, R0]
    resultado_antes = odeint(SIR_linea_base, y0, t1, args=(beta, gamma))

    # 2. Aplicar el refuerzo de vacunación
    S, I, R = resultado_antes[-1]
    print(f"Refuerzo aplicado el día {dia_vac}")
    print(f"Antes: S = {S:.2f}, R = {R:.2f}")
    vacunados = 0.5 * S
    S -= vacunados
    R += vacunados
    print(f"Después: S = {S:.2f}, R = {R:.2f}")

    # 3. Simulación desde el día del refuerzo hasta el final
    t2 = np.linspace(dia_vac, dias, dias - dia_vac + 1)
    resultado_despues = odeint(SIR_linea_base, [S, I, R], t2, args=(beta, gamma))

     # 4. Unir los dos resultados (evitando duplicar el día del refuerzo)
    resultado_completo = np.vstack((resultado_antes, resultado_despues[1:]))
    return resultado_completo, dia_vac

def graficar_resultado(t, resultado, titulo, dia_vac):
    """
    Función para gráficar.
    """
    S, I, R = resultado.T
    plt.figure(figsize=(10,6))
    plt.plot(t, S, 'b', label='Susceptibles')
    plt.plot(t, I, 'r', label='Infectados')
    plt.plot(t, R, 'g', label='Recuperados/Vacunados')

    if dia_vac is not None:
        plt.axvline(dia_vac, color='k', linestyle='--', label=f'Vacunación (día {dia_vac})')

    plt.title(titulo)
    plt.xlabel('Días')
    plt.ylabel('Personas')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

resultado, dia_vac = simular_con_vacunacion()
graficar_resultado(t, resultado, 'SIR con refuerzo único cuando $R_t > 1$', dia_vac)

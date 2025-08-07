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
t = np.linspace(0, dias, dias)

def SIR_linea_base(y, t, beta, gamma):
    """
    Modelo SIR básico sin intervenciones (línea base)
    """
    S, I, R = y
    
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    
    return [dSdt, dIdt, dRdt]

def simular_linea_base():
    """
    Simula el escenario de línea base
    """
    y0 = [S0, I0, R0]
    resultado = odeint(SIR_linea_base, y0, t, args=(beta, gamma))
    return resultado

def graficar_linea_base(t, resultado):
    """
    Grafica los resultados del escenario línea base
    """
    S, I, R = resultado.T
    
    plt.figure(figsize=(12, 8))
    
    # Gráfico principal
    plt.subplot(2, 1, 1)
    plt.plot(t, S, 'b-', label='Susceptibles', linewidth=2)
    plt.plot(t, I, 'r-', label='Infectados', linewidth=2)
    plt.plot(t, R, 'g-', label='Recuperados', linewidth=2)
    
    plt.xlabel('Días')
    plt.ylabel('Número de personas')
    plt.title('Línea Base: Modelo SIR sin intervenciones')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico enfocado en infectados
    plt.subplot(2, 1, 2)
    plt.plot(t, I, 'r-', label='Infectados', linewidth=2)
    plt.xlabel('Días')
    plt.ylabel('Infectados')
    plt.title('Serie temporal de infectados - Línea Base')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar estadísticas finales
    print(f"=== RESULTADOS LÍNEA BASE ===")
    print(f"Infectados máximos: {int(max(I)):,}")
    print(f"Día del pico: {int(t[np.argmax(I)])}")
    print(f"Total de infectados al final: {int(R[-1]):,}")
    print(f"Susceptibles restantes: {int(S[-1]):,}")
    print(f"Porcentaje de población infectada: {R[-1]/N*100:.1f}%")

# Ejecutar simulación
print("Simulando escenario línea base...")
resultado_base = simular_linea_base()
graficar_linea_base(t, resultado_base)
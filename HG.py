from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

'''
AI Disclaimer: ChatGPT was used to help generate the plotting and visualization
functions in this file. All physics, equations, and code logic were written
by me
'''

def magnetization(t, m,alpha,J,z, Tk):
    dm_dt = -alpha*(m - np.tanh((J*z*m)/(Tk)))
    return dm_dt

def magnetization_with_field(t, m, alpha, J, z, Tk,B0, omega):
    B_t = B0*np.sin(omega*t)
    dm_dt = -alpha*(m - np.tanh((J*z*m+B_t)/Tk))
    return dm_dt

def solve_magnetization(alpha,J, z, Tk, m0, t_span):
    sol = solve_ivp(magnetization, t_span,[m0], args=(alpha,J,z,Tk))
    return sol.t, sol.y[0]

def plot_solution(t, m, Tk):
    plt.figure(figsize=(10,6))
    plt.plot(t, m, label=f'k*Temperature = {Tk}')
    plt.title('Magnetization Over Time')
    plt.xlabel('Time')
    plt.ylabel('Magnetization m(t)')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.legend()
    plt.grid()
    plt.show()

def sweep_temperatures_overlay(alpha, J, z, Tk_range, m0, t_span):
    plt.figure(figsize=(10,6))
    for Tk in Tk_range:
        t, m_sol = solve_magnetization(alpha, J, z, Tk, m0, t_span)
        plt.plot(t, m_sol, label=f'T = {Tk}')
    plt.title('Magnetization Over Time for Different k*Temperatures')
    plt.xlabel('Time')
    plt.ylabel('Magnetization m(t)')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axhline(0.8, color='red', lw=0.5, ls='--', alpha=0.5)
    
    for Tk in Tk_range:
        t, m_sol = solve_magnetization(alpha, J, z, Tk, m0, t_span)
        crosses = np.where(m_sol >= 0.99)[0]
        if len(crosses) > 0:
            t_cross = t[crosses[0]]
            plt.axvline(t_cross, color='red', lw=2, ls='--', alpha=0.8)
            plt.text(t_cross, 0.5, f'Time: {t_cross:.2f}s', color='red', fontsize=10, ha='center')
            break
    
    plt.legend()
    plt.grid()
    plt.show()



def predator_response(alpha, J, z, Tk, m0, t_span, B0, omega):
    sol = solve_ivp(magnetization_with_field, t_span, [m0], args=(alpha, J, z, Tk, B0, omega))
    return sol.t, sol.y[0]

def plot_predator_response(t, m, B0, omega):
    plt.figure(figsize=(10,6))
    plt.plot(t, m, label=f'B0={B0}, ω={omega}')
    plt.title('Magnetization Response to Time-Varying Field')
    plt.xlabel('Time')
    plt.ylabel('Magnetization m(t)')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.legend()
    plt.grid()
    plt.show()


def sweep_B0_overlay(alpha, J, z, Tk, B0_range, m0, t_span, omega):
    plt.figure(figsize=(10, 6))
    for B0 in B0_range:
        t, m_sol = predator_response(alpha, J, z, Tk, m0, t_span, B0=B0, omega=omega)
        plt.plot(t, m_sol, label=f'B0 = {B0}')

    plt.title(f'Magnetization Over Time for Different B0 (ω = {omega})')
    plt.xlabel('Time')
    plt.ylabel('Magnetization m(t)')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.legend()
    plt.grid()
    plt.show()

def sweep_omega_overlay(alpha, J, z, Tk, B0, omega_range, m0, t_span):
    plt.figure(figsize=(10, 6))
    for omega in omega_range:
        t, m_sol = predator_response(alpha, J, z, Tk, m0, t_span, B0=B0, omega=omega)
        plt.plot(t, m_sol, label=f'ω = {omega}')

    plt.title(f'Magnetization Over Time for Different ω (B0 = {B0})')
    plt.xlabel('Time')
    plt.ylabel('Magnetization m(t)')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    alpha = 10.0
    J = 1.0
    z = 6
    m0 = 0.01
    t_span = (0, 5)
    Tk = 0.01
    B0 = 10.0
    omega = 2 * np.pi * 0.5

    omega_values = [1.0, 10, 100, 1000]
    B0_values = [1.0, 10, 100, 1000]
    sweep_omega_overlay(alpha, J, z, Tk, B0, omega_values, m0, t_span)
    #sweep_B0_overlay(alpha, J, z, Tk, B0_values, m0, t_span, omega)
import numpy as np
import matplotlib.pyplot as plt

T_values = np.linspace(300, 700, 9)

coefficients = {
    "CuO(s3)": [1.99377703E+00, 1.72262231E-02, -4.05047440E-05, 4.40904488E-08, -1.76268119E-11, -1.64454433E+03, -5.55887275E+00],
    "Cu(s3)": [1.60044064E+00, 3.44756251E-03, -3.83231798E-06, 1.40142396E-09, 6.89087125E-13, -2.50517802E+04, -1.05925277E+01],
    "H2": [0.03298124E+02, 0.08249442E-02, -0.08143015E-05, -0.09475434E-09, 0.04134872E-11, -0.01012521E+05, -0.03294094E+02],
    "CuH2O(s2)": [2.38076842E+00, 3.11178274E-02, -7.52443107E-05, 8.29419519E-08, -3.33949544E-11, -3.88170322E+04, -7.33530616E+00],
    "Cu(s2)": [1.76672074E+00, 7.34699433E-03, -1.54712960E-05, 1.50539592E-08, -5.24861336E-12, -7.43882087E+02, -7.70454044E+00],
}

def calculate_enthalpy(species, T):
    a = coefficients[species]
    H_RT = a[0] + a[1] * T / 2 + a[2] * T**2 / 3 + a[3] * T**3 / 4 + a[4] * T**4 / 5 + a[5] / T
    return H_RT * 8.314 * T  # Дж/моль

enthalpy_reactants = [calculate_enthalpy("CuO(s3)", T) + calculate_enthalpy("Cu(s3)", T) + calculate_enthalpy("H2", T) for T in T_values]
enthalpy_products = [calculate_enthalpy("CuH2O(s2)", T) + calculate_enthalpy("Cu(s2)", T) for T in T_values]

reaction_enthalpy = np.array(enthalpy_products) - np.array(enthalpy_reactants)

given_mean_enthalpy = -102698  # Дж/моль

calculated_mean_enthalpy = np.mean(reaction_enthalpy)

print("\nСредняя рассчитанная энтальпия реакции:", f"{calculated_mean_enthalpy:.2f} Дж/моль")
print("Заданное среднее значение:", f"{given_mean_enthalpy} Дж/моль")

plt.figure(figsize=(8, 5))
plt.plot(T_values, reaction_enthalpy, marker='o', linestyle='-', label='Рассчитанная энтальпия')
plt.axhline(given_mean_enthalpy, color='r', linestyle='--', label=f'Данное среднее: {given_mean_enthalpy:.2f} Дж/моль')
plt.axhline(calculated_mean_enthalpy, color='g', linestyle='--', label=f'Вычисленное среднее: {calculated_mean_enthalpy:.2f} Дж/моль')
plt.xlabel("Температура, K")
plt.ylabel("Энтальпия реакции, Дж/моль")
plt.title("Сравнение заданного и вычисленного среднего значения энтальпии реакции")
plt.legend()
plt.grid()
plt.show()

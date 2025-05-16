# %%
import numpy as np
from scipy.optimize import root_scalar
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from itertools import combinations, permutations

# %% [markdown]
# acceleration phase

# %%
def cumtrapz_like(y, x):
    """
    Mimics scipy.integrate.cumtrapz(y, x, initial=0)
    Computes cumulative trapezoidal integration.
    Returns an array of same length as y.
    """
    y = np.asarray(y)
    x = np.asarray(x)
    cumulative = np.zeros_like(y, dtype=float)

    for i in range(1, len(y)):
        dx = x[i] - x[i - 1]
        avg_height = 0.5 * (y[i] + y[i - 1])
        cumulative[i] = cumulative[i - 1] + dx * avg_height

    return cumulative

def get_rider_info(num, df):
    #athlete number
    athlete_name = f'M{num}'
    athlete = df[df['Name'] == athlete_name]

    W_prime = athlete["W'"].iloc[0]
    W_prime = W_prime * 1000 #convert to J
    CP = athlete['CP'].iloc[0]
    AC = athlete['CdA'].iloc[0]
    Pmax = athlete['Pmax'].iloc[0]
    m_rider = athlete['Mass'].iloc[0] #kg

    return W_prime, CP, AC, Pmax, m_rider

def simulate_accel_phase_with_thalf(s, P_const, num_of_half_laps, m_rider, m_wheels, P_init, v0, CdA, CP, rho=1.225, dt=0.05):
    track_half_lap = 125
    M = m_rider + m_wheels
    drag_coeff = 0.5 * rho * CdA

    # --- Phase 1: Increasing power ---
    t_vals1 = [0]
    v_vals1 = [v0]
    x = 0
    t = 0
    v = v0

    while x < track_half_lap:
        P = P_init + s * t
        drag = drag_coeff * v**3 
        a = (P - drag) / (M * v) if v > 0 else 0.0
        v += a * dt
        x += v * dt
        t += dt
        t_vals1.append(t)
        v_vals1.append(v)

    t_half = t_vals1[-1]
    v_half = v_vals1[-1]

    # --- Phase 2: Constant power ---
    t_vals2 = [t_half]
    v_vals2 = [v_half]
    x2 = 0
    v = v_half
    t = t_half

    remaining_dist = (num_of_half_laps - 1) * track_half_lap

    while x2 < remaining_dist:
        drag = 0.5 * rho * v**3 * CdA
        a = (P_const - drag) / (M * v) if v > 0 else 0.0
        v += a * dt
        x2 += v * dt
        t += dt
        t_vals2.append(t)
        v_vals2.append(v)

    t_fin = t_vals2[-1]
    v_final = v_vals2[-1]

    # --- Combine phase 1 and phase 2 ---
    t_total = np.array(t_vals1[:-1] + t_vals2)  # remove duplicate t_half
    v_total = np.array(v_vals1[:-1] + v_vals2)

    P_total = np.where(t_total <= t_half, P_init + s * t_total, P_const)

    over_CP = P_total - CP
    over_CP[over_CP < 0] = 0
    Wprime_used = np.trapezoid(over_CP, t_total)

    return t_fin, Wprime_used, v_final, t_half, t_total, v_total


# Updated optimizer to return t_half
def find_best_power_profile(s_range, P_bounds, num_of_half_laps, v_target, m_rider, m_wheels, P_init, v0, CdA, CP, epsilon=0.4, rho=1.225):
    # start_time2 = time.time()
    best_result = None
    min_Wprime = float('inf')

    for s in s_range:
        cached_result = {}

        def v_error(P):
            try:
                result = simulate_accel_phase_with_thalf(s, P, num_of_half_laps,  m_rider, m_wheels, P_init, v0, CdA, CP, rho)
                cached_result["result"] = result  # store it to avoid recomputation
                _, _, v_final, _, _, _ = result
                return v_final - v_target
            except Exception:
                cached_result["result"] = None
                return np.inf

        try:
            root = root_scalar(v_error, bracket=P_bounds, method='bisect', xtol=1e-3, rtol=1e-3)
        except ValueError:
            continue

        if not root.converged or cached_result["result"] is None:
            continue

        # Use cached result instead of re-simulating
        tfin, Wprime_used, v_final, t_half, t_total, v_total = cached_result["result"]

        if abs(v_final - v_target) < epsilon and Wprime_used < min_Wprime:
            best_result = {
                's': s,
                'P_const': root.root,
                'tfin': tfin,
                'Wprime_used': Wprime_used,
                'v_final': v_final,
                't_half': t_half,
                't_array': t_total,
                'v_array': v_total
            }
            min_Wprime = Wprime_used

    # end_time2 = time.time()
    # print(f"Optimization time: {end_time2 - start_time2:} seconds")
    return best_result

def accel_phase(v0, P0, v_target, chosen_athletes, start_order, drafting_percents, df, acc_half_laps, bank_angle, rho=1.225, m_wheels=0.75, g = 9.81):
    rider_data = {}
    W_rem = {}

    # Build rider data and initial W'
    for rider in chosen_athletes:
        W_prime, CP, AC, Pmax, m_rider = get_rider_info(rider, df)
        rider_data[rider] = {
            "W_prime": W_prime,
            "CP": CP,
            "AC": AC,
            "Pmax": Pmax,
            "m_rider": m_rider,
        }
        W_rem[rider] = W_prime
    
    leader = start_order[0]

    sweep_s = np.linspace(50, 150, 10)     # Sweep slopes from 
    P_bounds = (400, 750)                  # Reasonable range for constant power

    best_power_profile = find_best_power_profile(sweep_s, P_bounds, acc_half_laps, v_target, rider_data[leader]["m_rider"], m_wheels, P0, v0, rider_data[leader]["AC"], rider_data[leader]["CP"], rho)
    if best_power_profile is None:
        raise ValueError(f"No feasible acceleration found for target velocity {v_target:.2f} m/s.")
    slope = best_power_profile['s']
    P_const = best_power_profile['P_const']
    tfin = best_power_profile['tfin']
    wprime_dec1 = best_power_profile['Wprime_used']
    t_half = best_power_profile['t_half']

    #what the velo profile looks like
    t_sim = best_power_profile['t_array']
    v_sim = best_power_profile['v_array']

    # power function
    P_model = np.piecewise(
    t_sim,
    [t_sim <= (t_half), t_sim > (t_half)],
    [lambda t: P0 + slope * (t - 5), P_const])

    # solve for the acceleration
    _, unique_indices = np.unique(t_sim, return_index=True)
    t_sim_clean = t_sim[np.sort(unique_indices)]
    v_sim_clean = v_sim[np.sort(unique_indices)]
    a_sim = np.gradient(v_sim_clean, t_sim_clean)
    # a_sim = np.gradient(v_sim, t_sim)  # derivative dv/dt
    P_model_clean = P_model[np.sort(unique_indices)]

    # power profile for other riders
    power2 = rider_data[start_order[1]]["m_rider"]/rider_data[leader]["m_rider"]*P_model_clean-1/2*rho*(v_sim_clean**3)*(rider_data[leader]["AC"]/(rider_data[leader]["m_rider"])-drafting_percents[1]*rider_data[start_order[1]]["AC"]/(rider_data[start_order[1]]["m_rider"]))
    power3 = rider_data[start_order[2]]["m_rider"]/rider_data[leader]["m_rider"]*P_model_clean-1/2*rho*(v_sim_clean**3)*(rider_data[leader]["AC"]/(rider_data[leader]["m_rider"])-drafting_percents[2]*rider_data[start_order[2]]["AC"]/(rider_data[start_order[2]]["m_rider"]))
    power4 = rider_data[start_order[3]]["m_rider"]/rider_data[leader]["m_rider"]*P_model_clean-1/2*rho*(v_sim_clean**3)*(rider_data[leader]["AC"]/(rider_data[leader]["m_rider"])-drafting_percents[3]*rider_data[start_order[3]]["AC"]/(rider_data[start_order[3]]["m_rider"]))

    # computing the energy for each rider
    energy2 = np.trapezoid(power2, t_sim_clean) - rider_data[start_order[1]]["CP"]*(t_sim_clean[-1]) - rider_data[start_order[1]]["m_rider"]*g*np.sin(bank_angle)
    energy3 = np.trapezoid(power3, t_sim_clean) - rider_data[start_order[2]]["CP"]*(t_sim_clean[-1]) - rider_data[start_order[2]]["m_rider"]*g*2*np.sin(bank_angle)
    energy4 = np.trapezoid(power4, t_sim_clean) - rider_data[start_order[3]]["CP"]*(t_sim_clean[-1]) - rider_data[start_order[3]]["m_rider"]*g*3*np.sin(bank_angle)

    # updating W' for all riders
    W_rem[leader] -= wprime_dec1
    W_rem[start_order[1]] -= energy2 
    W_rem[start_order[2]] -= energy3
    W_rem[start_order[3]] -= energy4

    # return tfin, W_rem, t_sim, v_sim, slope, P_const, t_half, a_sim
    return rider_data, tfin, W_rem, t_sim_clean, v_sim_clean, slope, P_const, t_half, a_sim

# %% [markdown]
# constant velocity part (from assume_constant_velocity)

# %%
# 1. Put switch schedule in easier format (i.e. how many laps each person leads).
# Switch schedule is now 1x32 list of 1s and 0s. If there is a 1 at index i (starting at 0), that means we switch after i half laps. 
# This is intended to account for switching at the beginning of the steady-state phase (i.e. after zero half-laps).
def format_ss(ss):
    i = 0
    c = 0
    f_ss = []
    while i < len(ss):
        if ss[i] == 1:
            f_ss.append(c)
            c = 1
        else:
            c += 1
        i += 1
    f_ss.append(c)
    return f_ss

# 2/3. Calculate the energy used as a function of velocity. 
# E = P * t = P * d / v = (drag_adv * 0.5 * rho * CdA * v ** 3 + m * g * Crr * v - CP) * d / v = drag_adv * 0.5 * rho * CdA * d * v ** 2 + 
# m * g * Crr * d - CP * d / v
# Also need to check that we are within bounds of power curve: drag_adv * 0.5 * rho * CdA * v ** 3 + m * g * Crr * v = W' / t  CP = W' * v / d + CP
# drag_adv * 0.5 * rho * CdA * v ** 3 + (m * g * Crr - W' / d) * v - CP = 0
def phase(f_ss, rider_stats, drag_adv, order, rho = 1.225, Crr = 0.0018, g = 9.80665, bike_length = 2.1):
    i = 0
    energy = {rider: [0, 0, 0] for rider in order}
    # v_max = float("inf")
    while i < len(f_ss):
        if i == 0: # lose a bike length for every switch
            penalty = 0
        else:
            penalty = bike_length
        for pos, rider in enumerate(order):
            if pos == 0 and i < len(f_ss) - 1: # have to maintain lead power for a quarter lap
                quarter_lap = 250 / 4
            else:
                quarter_lap = 0
            d = f_ss[i] * 125 + quarter_lap + penalty
            energy[rider][0] += drag_adv[pos] * 0.5 * rho * rider_stats[rider]["AC"] * d
            energy[rider][1] += rider_stats[rider]["m_rider"] * g * Crr * d
            energy[rider][2] += rider_stats[rider]["CP"] * d
        # Find maximum velocity within power curve by checking power output of lead riders
        # lead_d = f_ss[i] * 125 + penalty
        # if i < len(f_ss) - 1:
        #     lead_d += 250 / 4
        # if lead_d == 0:
        #     pass
        # else:
        #     pc_roots = np.roots([drag_adv[0] * 0.5 * rho * rider_stats[order[0]]["CdA"], 0, 
        #                               rider_stats[order[0]]["mass"] * g * Crr - rider_stats[order[0]]["W'"] / lead_d, -rider_stats[order[0]]["CP"]])
        #     v = np.real(max([root for root in pc_roots if np.isreal(root)]))
        #     v_max = min(v, v_max)
        order = order[1:] + order[:1]
        i += 1
    return energy, order

# 2/3. If peeling, calculate energy usage before and after peel. If not, calculate energy usage for whole race. Assumes that peel takes place at a
#      switch, but does not explicitly check.
def race(peel, f_ss, rider_stats, drag_adv, order = [1,2,3,4], rho = 1.225, Crr = 0.0018, g = 9.80665, bike_length = 2.1):
    if peel:
        f_ss1 = []
        half_laps = 0
        i = 0
        while half_laps < peel:
            f_ss1.append(f_ss[i])
            half_laps += f_ss[i]
            i += 1
        f_ss2 = f_ss[i:]
        energy1, order1 = phase(f_ss1, rider_stats, drag_adv, order, rho, Crr, g, bike_length)
        energy2, order2 = phase(f_ss2, rider_stats, drag_adv, order1[:-1], rho, Crr, g, bike_length) # We have already executed the switch in the order, so get rid of the last rider
        energy2[order1[-1]] = [0, 0, 0]
        energy = {rider: [energy1[rider][i] + energy2[rider][i] for i in range(3)] for rider in energy1}
        return energy
    return phase(f_ss, rider_stats, drag_adv, [1,2,3,4], rho, Crr, g, bike_length)[0]


# 4. Solve the constraint equations a * v^2 + b - c / v < W'
def max_v(energy, rider_stats):
    v = float("inf")
    for rider in energy:
        velos = np.roots([energy[rider][0], 0, energy[rider][1] - rider_stats[rider]["W_prime"], -energy[rider][2]])
        velo = np.real(max([root for root in velos if np.isreal(root)]))
        v = min(v,velo)
    return v

# 5. Do the whole process and solve t = d / v
def find_time(peel, ss, rider_stats, drag_adv, order = [1,2,3,4], rho = 1.225, Crr = 0.0018, g = 9.80665, bike_length = 2.1):
    f_ss = format_ss(ss)
    energy = race(peel, f_ss, rider_stats, drag_adv, order, rho, Crr, g, bike_length)
    v = max_v(energy,rider_stats)
    return 4000 / v

# Take velocity as input, calculate total energy expenditure in a phase directly
# E = P * t = P * d / v = (drag_adv * 0.5 * rho * CdA * v ** 3 + m * g * Crr * v - CP) * d / v = drag_adv * 0.5 * rho * CdA * d * v ** 2 + 
# m * g * Crr * d - CP * d / v
def phase_energy(vel, f_ss, rider_stats, drag_adv, order, rho = 1.225, Crr = 0.0018, g = 9.80665, bike_length = 2.1):
    i = 0
    energy = {rider: 0 for rider in order}
    # v_max = float("inf")
    while i < len(f_ss):
        if i == 0: # lose a bike length for every switch
            penalty = 0
        else:
            penalty = bike_length
        for pos, rider in enumerate(order):
            if pos == 0 and i < len(f_ss) - 1: # have to maintain lead power for a quarter lap
                quarter_lap = 250 / 4
            else:
                quarter_lap = 0
            energy[rider] += (drag_adv[pos] * 0.5 * rho * rider_stats[rider]["AC"] * vel ** 2 + 
                              rider_stats[rider]["m_rider"] * g * Crr - rider_stats[rider]["CP"] / vel) * (f_ss[i] * 125 + quarter_lap + penalty)
        # Find maximum velocity within power curve by checking power output of lead riders
        # lead_d = f_ss[i] * 125 + penalty
        # if i < len(f_ss) - 1:
        #     lead_d += 250 / 4
        # if lead_d == 0:
        #     pass
        # else:
        #     pc_roots = np.roots([drag_adv[0] * 0.5 * rho * rider_stats[order[0]]["CdA"], 0, 
        #                               rider_stats[order[0]]["mass"] * g * Crr - rider_stats[order[0]]["W'"] / lead_d, -rider_stats[order[0]]["CP"]])
        #     v = np.real(max([root for root in pc_roots if np.isreal(root)]))
        #     v_max = min(v, v_max)
        order = order[1:] + order[:1]
        i += 1
    return energy, order

def race_energy(vel, peel, switch_schedule, rider_stats, drag_adv, order = [1,2,3,4], rho = 1.225, Crr = 0.0018, g = 9.80665, bike_length = 2.1):
    f_ss = format_ss(switch_schedule)
    if peel:
        f_ss1 = []
        half_laps = 0
        i = 0
        while half_laps < peel:
            f_ss1.append(f_ss[i])
            half_laps += f_ss[i]
            i += 1
        f_ss2 = f_ss[i:]
        energy1, order1 = phase_energy(vel, f_ss1, rider_stats, drag_adv, order, rho, Crr, g, bike_length)
        energy2, order2 = phase_energy(vel, f_ss2, rider_stats, drag_adv, order1[:-1], rho, Crr, g, bike_length) # We have already executed the switch in the order, so get rid of the last rider
        energy2[order1[-1]] = 0
        energy = {rider: energy1[rider] + energy2[rider] for rider in energy1}
        return energy, order2
    return phase_energy(vel, f_ss, rider_stats, drag_adv, [1,2,3,4], rho, Crr, g, bike_length)[0]

# %% [markdown]
# now combining them

# %%
def combined(acc_func, ss_func, peel, switch_schedule, drag_adv, df, 
             chosen_athletes=[1,2,3,4], order=[1,2,3,4], 
             min_v=15, max_v=22, precision=200, acc_length=4, 
             rho=1.225, Crr=0.0018, g=9.80665, bike_length=2.1, 
             m_wheels=0.75, v0=1.5, P0=500, bank_angle=np.radians(12)):

    while True:
        v = (min_v + max_v) / 2
        print(f"Trying v: {v}")
        try:
            rider_data, tfin, W_rem, _, _, slope, P_const, t_half_lap, _ = acc_func(
                v0, P0, v, chosen_athletes, order, drag_adv, df, acc_length, bank_angle, rho, m_wheels, g
            )
        except ValueError:
            # If no feasible acceleration is found, treat v as too high
            max_v = v
            continue

        ss_energy, final_order = ss_func(v, peel - acc_length, switch_schedule[acc_length:], rider_data, drag_adv, order, rho, Crr, g, bike_length)
        
        errors = [W_rem[rider] - ss_energy[rider] for rider in order]

        if any(error < 0 for error in errors):
            max_v = v
        elif any(error < precision for error in errors):
            t_tot = tfin + (32 - acc_length) * 125 / v
            return v, t_tot, errors, slope, P_const, t_half_lap, final_order
        elif abs(max_v - min_v) < 0.005:
            # If the difference between max_v and min_v is very small, return the current v
            t_tot = tfin + (32 - acc_length) * 125 / v
            return v, t_tot, errors, slope, P_const, t_half_lap, final_order
        else:
            min_v = v

# %% [markdown]

## Uncoment this for a test run 

# # %%
# #reading in data from spreadsheet
# df = pd.read_excel('final_data_sheet.xlsx')
# # Load the data
# power_curve = pd.read_excel("final_data_sheet.xlsx", sheet_name="Power Curves")

# rider_data = {}
# W_rem = {}
# chosen_athletes = [1, 2, 3, 4]  # Example athlete numbers

# for rider in chosen_athletes:
#     W_prime, CP, AC, Pmax, m_rider = get_rider_info(rider, df)
#     rider_data[rider] = {
#         "W_prime": W_prime,
#         # "t_prime": t_prime,
#         "CP": CP,
#         "AC": AC,
#         # "Pmax": Pmax,
#         "m_rider": m_rider,
#     }
#     W_rem[rider] = W_prime

# drag_adv = [1, 0.58, 0.52, 0.53]
# switch_schedule = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# switch_schedule2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# switch_schedule3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
# switch_schedule4 = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]

# starttime = time.time()
# v_SS, t_final, W_rem, slope, P_const, t_half_lap, final_order = combined(
#     accel_phase,
#     race_energy,
#     25,
#     switch_schedule4,
#     drag_adv,
#     df,
#     order=[1, 2, 3, 4]
# )
# endtime = time.time()
# print(f"Time taken: {endtime - starttime:.2f} seconds")

# # %%
# print(f"Steady State Velocity: {v_SS:.2f} m/s")
# print(f"Total time: {t_final:.2f} seconds")
# print(f"Remaining W': {W_rem}")
# print(f"Slope: {slope:.2f} W/m")
# print(f"Constant Power: {P_const:.2f} W")
# print(f"Time to reach half lap: {t_half_lap:.2f} seconds")



# %%
import numpy as np
from scipy.optimize import root_scalar
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from itertools import combinations, permutations

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from matplotlib.table import Table


# %% [markdown]
# acceleration phase

# %%

# def get_rider_info(num, df):
#     #athlete number
#     athlete_name = f'M{num}'
#     athlete = df[df['Name'] == athlete_name]

#     W_prime = athlete["W'"].iloc[0]
#     W_prime = W_prime * 1000 #convert to J
#     t_prime = athlete['Tmax'].iloc[0]
#     CP = athlete['CP'].iloc[0]
#     AC = athlete['CdA'].iloc[0]
#     Pmax = athlete['Pmax'].iloc[0]
#     m_rider = athlete['Mass'].iloc[0] #kg

#     return W_prime, t_prime, CP, AC, Pmax, m_rider
def get_rider_info(num, df, number_to_name):
    name = number_to_name[num]
    athlete = df[df['Name'] == name]

    W_prime = athlete["W'"].iloc[0] * 1000  # J
    CP = athlete['CP'].iloc[0]
    AC = athlete['CdA'].iloc[0]
    Pmax = athlete['Pmax'].iloc[0]
    m_rider = athlete['Mass'].iloc[0]

    return W_prime, CP, AC, Pmax, m_rider


def simulate_accel_phase_with_thalf(s, P_const, num_of_half_laps, m_rider, m_wheels, P_init, v0, CdA, CP, Crr=0.0018, rho=1.225, dt=0.05):
    track_half_lap = 125
    M = m_rider + m_wheels
    drag_coeff = 0.5 * rho * CdA

    # --- Phase 1: Increasing power ---
    t_vals1 = [0]
    v_vals1 = [v0]
    x = 0
    t = 0
    v = v0

    while x < track_half_lap*3/2:
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

    # remaining_dist = (num_of_half_laps - 1) * track_half_lap
    remaining_dist = (num_of_half_laps - 1.5) * track_half_lap

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
                # print(f"s={s:.1f}, P={P:.1f}, v_final={v_final:.2f}, target={v_target:.2f}")
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

def accel_phase(v0, P0, Pmax, v_target, start_order, drafting_percents, df, acc_half_laps, bank_angle, rider_data, W_rem_start, rho=1.225, m_wheels=0.75, g = 9.81):
    # rider_data = {}
    W_rem = W_rem_start.copy()
    
    leader = start_order[0]

    sweep_s = np.linspace(50, 90, 3)     # Sweep slopes from 
    P_bounds = (400, Pmax)                  # Reasonable range for constant power

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
    P_model = P0 + slope * (t_sim)
    P_model[t_sim > t_half] = P_const


    # solve for the acceleration
    _, unique_indices = np.unique(t_sim, return_index=True)
    t_sim_clean = t_sim[np.sort(unique_indices)]
    v_sim_clean = v_sim[np.sort(unique_indices)]
    a_sim = np.gradient(v_sim_clean, t_sim_clean)
    # a_sim = np.gradient(v_sim, t_sim)  # derivative dv/dt
    P_model_clean = P_model[np.sort(unique_indices)]

    # power profile for other riders
    v3 = v_sim_clean ** 3

    m_leader = rider_data[leader]["m_rider"]
    AC_leader = rider_data[leader]["AC"]
    AC_m_leader = AC_leader / m_leader

    m2 = rider_data[start_order[1]]["m_rider"]
    m3 = rider_data[start_order[2]]["m_rider"]
    m4 = rider_data[start_order[3]]["m_rider"]

    AC_m2 = rider_data[start_order[1]]["AC"] / m2
    AC_m3 = rider_data[start_order[2]]["AC"] / m3
    AC_m4 = rider_data[start_order[3]]["AC"] / m4

    power2 = (m2 / m_leader) * P_model_clean - 0.5 * rho * v3 * (AC_m_leader - drafting_percents[1] * AC_m2)
    power3 = (m3 / m_leader) * P_model_clean - 0.5 * rho * v3 * (AC_m_leader - drafting_percents[2] * AC_m3)
    power4 = (m4 / m_leader) * P_model_clean - 0.5 * rho * v3 * (AC_m_leader - drafting_percents[3] * AC_m4)



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
    return tfin, W_rem, t_sim_clean, v_sim_clean, slope, P_const, t_half, a_sim

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
    num_riders = len(order)
    num_changes = len(f_ss)
    # v_max = float("inf")
    while i < num_changes:
        if i == 0: # lose a bike length for every switch
            penalty = 0
        else:
            penalty = bike_length
        for pos, rider in enumerate(order):
            if pos == 0 and i < num_changes - 1: # have to maintain lead power for a quarter lap
                quarter_lap = 250 / 4
            elif pos == num_riders - 1 and i > 0: # already adding extra quarter lap of leading power from previous cycle
                quarter_lap = -250 / 4
            else:
                quarter_lap = 0
            d = f_ss[i] * 125 + quarter_lap + penalty
            energy[rider][0] += drag_adv[pos] * 0.5 * rho * rider_stats[rider]["AC"] * d
            energy[rider][1] += rider_stats[rider]["m_rider"] * g * Crr * d
            energy[rider][2] += rider_stats[rider]["CP"] * d

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
def phase_energy(vel, f_ss, rider_stats, drag_adv, order, end, rho = 1.225, Crr = 0.0018, g = 9.80665, bike_length = 2.1, m_sys = 10.0):
    i = 0
    energy = {rider: 0 for rider in order}
    num_riders = len(order)
    num_changes = len(f_ss)
    # v_max = float("inf")
    while i < num_changes:
        if i == 0: # lose a bike length for every switch
            penalty = 0
        else:
            penalty = bike_length
        for pos, rider in enumerate(order):
            if pos == 0 and i < num_changes - 1: # have to maintain lead power for a quarter lap
                quarter_lap = 250 / 4
            elif pos == num_riders - 1 and i > 0: # already adding extra quarter lap of leading power from previous cycle
                quarter_lap = -250 / 4
            else:
                quarter_lap = 0
            if end and i == num_changes - 1:
                last_lap = -250 / 4
            else:
                last_lap = 0
            # energy[rider] += (drag_adv[pos] * 0.5 * rho * rider_stats[rider]["AC"] * vel ** 2 + (rider_stats[rider]["m_rider"] + m_sys) * g * Crr - rider_stats[rider]["CP"] / vel) * (f_ss[i] * 125 + quarter_lap + penalty + last_lap)
            energy[rider] = max(0, energy[rider] + (drag_adv[pos] * 0.5 * rho * rider_stats[rider]["AC"] * vel ** 2 + (rider_stats[rider]["m_rider"] + m_sys) * g * Crr - rider_stats[rider]["CP"] / vel) * (f_ss[i] * 125 + quarter_lap + penalty + last_lap))

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
        energy1, order1 = phase_energy(vel, f_ss1, rider_stats, drag_adv, order, False, rho, Crr, g, bike_length)
        energy2, order2 = phase_energy(vel, f_ss2, rider_stats, drag_adv, order1[:-1], True, rho, Crr, g, bike_length) # We have already executed the switch in the order, so get rid of the last rider
        energy2[order1[-1]] = 0
        energy = {rider: energy1[rider] + energy2[rider] for rider in energy1}
        return energy
    return phase_energy(vel, f_ss, rider_stats, drag_adv, order, True, rho, Crr, g, bike_length)[0]

# %% [markdown]
# now combining them

# %%
def combined(acc_func, ss_func, peel, switch_schedule, drag_adv, df, rider_data, W_rem,
             order=[1,2,3,4], 
             min_v=15, max_v=22, precision=200, acc_length=3, 
             rho=1.225, Crr=0.0018, g=9.80665, bike_length=2.1, 
             m_wheels=0.75, v0=1.5, P0=500, bank_angle=np.radians(12)):

    leader = order[0]

    while True:
        v = (min_v + max_v) / 2
        print(f"Trying v: {v}")
        try:
            tfin, W_rem_updated, _, _, slope, P_const, t_half_lap, _ = acc_func(
                v0, P0, rider_data[leader]["Pmax"],v, order, drag_adv, df, acc_length, bank_angle,W_rem_start = W_rem, rider_data=rider_data, rho=rho, m_wheels=m_wheels, g=g
            )
        except ValueError:
            # If no feasible acceleration is found, treat v as too high
            if abs(max_v - min_v) < 0.005:
                # If the difference between max_v and min_v is very small, return the current v
                return v, "valid accel phase not found", "valid accel phase not found", "valid accel phase not found", \
                    "valid accel phase not found", "valid accel phase not found"
            max_v = v
            continue

        ss_energy = ss_func(v, peel - acc_length, switch_schedule[acc_length:], rider_data, drag_adv, order, rho, Crr, g, bike_length)
        
        errors = [W_rem_updated[rider] - ss_energy[rider] for rider in order]
        # print(f"errors: {errors}")

        if any(error < 0 for error in errors):
            max_v = v
        elif any(error < precision for error in errors):
            t_tot = tfin + (32 - acc_length) * 125 / v
            return v, t_tot, errors, slope, P_const, t_half_lap
        elif abs(max_v - min_v) < 0.005:
            # If the difference between max_v and min_v is very small, return the current v
            t_tot = tfin + (32 - acc_length) * 125 / v
            return v, t_tot, errors, slope, P_const, t_half_lap
        else:
            min_v = v

# %% [markdown]
# calling this

# %%
#reading in data from spreadsheet
# df = pd.read_excel('2024_Performance_Physiology_Data_Anonymised_v2.xlsx')
df = pd.read_excel('final_data_sheet.xlsx')
# Load the data
# power_curve = pd.read_excel("2024_Performance_Physiology_Data_Anonymised_v2.xlsx", sheet_name="Power Curves")

name_to_number = {}
number_to_name = {}
rider_data = {}
W_rem = {}

chosen_names = df["Name"].tolist()
for i, name in enumerate(chosen_names, start=1):
    name_to_number[name] = i
    number_to_name[i] = name

chosen_athletes = [1, 2, 3, 4]  # Example athlete numbers

for rider in chosen_athletes:
    W_prime,  CP, AC, Pmax, m_rider = get_rider_info(rider, df, number_to_name)
    rider_data[rider] = {
        "W_prime": W_prime,
        "CP": CP,
        "AC": AC,
        "Pmax": Pmax,
        "m_rider": m_rider,
    }
    W_rem[rider] = W_prime

drag_adv = [1, 0.58, 0.52, 0.53]
switch_schedule = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
switch_schedule2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
switch_schedule3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
switch_schedule4 = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
order = [1, 2, 3, 4]

starttime = time.time()
print(race_energy(18, 29, switch_schedule4, rider_data, drag_adv))
v_SS, t_final, W_rem, slope, P_const, t_half_lap = combined(accel_phase, race_energy, 29, switch_schedule4, drag_adv, df, rider_data, W_rem, P0 = 50, order = order)
print(v_SS, t_final, W_rem, slope, P_const, t_half_lap)
endtime = time.time()
print(f"Time taken: {endtime - starttime:.2f} seconds")

# %%
print(f"Steady State Velocity: {v_SS:.2f} m/s")
print(f"Total time: {t_final:.2f} seconds")
print(f"Remaining W': {W_rem}")
print(f"Slope: {slope:.2f} W/m")
print(f"Constant Power: {P_const:.2f} W")
print(f"Time to reach half lap: {t_half_lap:.2f} seconds")

# %% [markdown]
# here is where we change stuff for the visuals

# %%
def accel_phase2(v0, P0, Pmax, v_target, start_order, drafting_percents, df, acc_half_laps, bank_angle, rider_data, W_rem_start, rho=1.225, m_wheels=0.75, g = 9.81):
    W_rem = W_rem_start.copy()
    power_profile_acc = {}
    
    leader = start_order[0]

    sweep_s = np.linspace(50, 90, 3)     # Sweep slopes from 
    P_bounds = (400, Pmax)                  # Reasonable range for constant power

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
    P_model = P0 + slope * (t_sim)
    P_model[t_sim > t_half] = P_const


    # solve for the acceleration
    _, unique_indices = np.unique(t_sim, return_index=True)
    t_sim_clean = t_sim[np.sort(unique_indices)]
    v_sim_clean = v_sim[np.sort(unique_indices)]
    a_sim = np.gradient(v_sim_clean, t_sim_clean)
    # a_sim = np.gradient(v_sim, t_sim)  # derivative dv/dt
    P_model_clean = P_model[np.sort(unique_indices)]
    power_profile_acc[leader] = P_model_clean

    # power profile for other riders
    v3 = v_sim_clean ** 3

    m_leader = rider_data[leader]["m_rider"]
    AC_leader = rider_data[leader]["AC"]
    AC_m_leader = AC_leader / m_leader

    m2 = rider_data[start_order[1]]["m_rider"]
    m3 = rider_data[start_order[2]]["m_rider"]
    m4 = rider_data[start_order[3]]["m_rider"]

    AC_m2 = rider_data[start_order[1]]["AC"] / m2
    AC_m3 = rider_data[start_order[2]]["AC"] / m3
    AC_m4 = rider_data[start_order[3]]["AC"] / m4

    power2 = (m2 / m_leader) * P_model_clean - 0.5 * rho * v3 * (AC_m_leader - drafting_percents[1] * AC_m2)
    power3 = (m3 / m_leader) * P_model_clean - 0.5 * rho * v3 * (AC_m_leader - drafting_percents[2] * AC_m3)
    power4 = (m4 / m_leader) * P_model_clean - 0.5 * rho * v3 * (AC_m_leader - drafting_percents[3] * AC_m4)
    power_profile_acc[start_order[1]] = power2
    power_profile_acc[start_order[2]] = power3
    power_profile_acc[start_order[3]] = power4


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
    return tfin, W_rem, t_sim_clean, v_sim_clean, slope, P_const, t_half, a_sim, power_profile_acc

# %%
# Take velocity as input, calculate total energy expenditure in a phase directly
# E = P * t = P * d / v = (drag_adv * 0.5 * rho * CdA * v ** 3 + m * g * Crr * v - CP) * d / v = drag_adv * 0.5 * rho * CdA * d * v ** 2 + 
# m * g * Crr * d - CP * d / v
def phase_energy2(vel, f_ss, rider_stats, drag_adv, order, end, race_powers, race_energies, total_energies, rho = 1.225, Crr = 0.0018, g = 9.80665, bike_length = 2.1, m_sys = 10.0):
    i = 0
    energy = {rider: 0 for rider in order}
    num_riders = len(order)
    num_changes = len(f_ss)
    # v_max = float("inf")
    while i < num_changes:
        if i == 0: # lose a bike length for every switch
            penalty = 0
        else:
            penalty = bike_length
        for pos, rider in enumerate(order):
            if pos == 0 and i < num_changes - 1: # have to maintain lead power for a quarter lap
                quarter_lap = 250 / 4
            elif pos == num_riders - 1 and i > 0: # already adding extra quarter lap of leading power from previous cycle
                quarter_lap = -250 / 4
            else:
                quarter_lap = 0
            if end and i == num_changes - 1:
                last_lap = -250 / 4
            else:
                last_lap = 0
            
            #W' for the segment
            # segment_energy = (drag_adv[pos] * 0.5 * rho * rider_stats[rider]["AC"] * vel ** 2 + (rider_stats[rider]["m_rider"] + m_sys) * g * Crr - rider_stats[rider]["CP"] / vel) * (f_ss[i] * 125 + quarter_lap + penalty + last_lap)
            segment_energy =  max(0, energy[rider] + (drag_adv[pos] * 0.5 * rho * rider_stats[rider]["AC"] * vel ** 2 + (rider_stats[rider]["m_rider"] + m_sys) * g * Crr - rider_stats[rider]["CP"] / vel) * (f_ss[i] * 125 + quarter_lap + penalty + last_lap))
            
            race_energies[rider].append(segment_energy)

            #actual energy expenditure
            segment_total_energy =  (drag_adv[pos] * 0.5 * rho * rider_stats[rider]["AC"] * vel ** 2 + (rider_stats[rider]["m_rider"] + m_sys) * g * Crr) * (f_ss[i] * 125 + quarter_lap + penalty + last_lap)
            total_energies[rider].append(segment_total_energy)

            #cumulative energy expenditure
            energy[rider] = segment_energy
            
            P_rider = (drag_adv[pos] * 0.5 * rho * rider_stats[rider]["AC"] * vel**3 + (rider_stats[rider]["m_rider"] + m_sys) * g * Crr * vel)
            race_powers[rider].append(P_rider)

        order = order[1:] + order[:1]
        i += 1

    return energy, order

def race_energy2(vel, peel, switch_schedule, rider_stats, drag_adv, order = [1,2,3,4], rho = 1.225, Crr = 0.0018, g = 9.80665, bike_length = 2.1):
    race_powers = {rider: [] for rider in order}
    race_energies = {rider: [] for rider in order}
    total_energies = {rider: [] for rider in order}
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
        energy1, order1 = phase_energy2(vel, f_ss1, rider_stats, drag_adv, order, False, race_powers, race_energies,total_energies,  rho, Crr, g, bike_length)
        energy2, order2 = phase_energy2(vel, f_ss2, rider_stats, drag_adv, order1[:-1], True, race_powers,race_energies, total_energies, rho, Crr, g, bike_length) # We have already executed the switch in the order, so get rid of the last rider
        energy2[order1[-1]] = 0
        energy = {rider: energy1[rider] + energy2[rider] for rider in energy1}
        return energy, race_powers, race_energies, total_energies
    return phase_energy2(vel, f_ss, rider_stats, drag_adv, order, True, rho, Crr, g, bike_length)[0], race_powers, race_energies, total_energies

# %%
def combined2(acc_func, ss_func, peel, switch_schedule, drag_adv, df, rider_data, W_rem,
             order=[1,2,3,4], 
             min_v=15, max_v=22, precision=200, acc_length=3, 
             rho=1.225, Crr=0.0018, g=9.80665, bike_length=2.1, 
             m_wheels=0.75, v0=1.5, P0=500, bank_angle=np.radians(12)):

    leader = order[0]

    while True:
        v = (min_v + max_v) / 2
        print(f"Trying v: {v}")
        try:
            tfin, W_rem_updated, _, v_sim_clean, slope, P_const, t_half_lap, _, power_profile_acc = acc_func(
                v0, P0, rider_data[leader]["Pmax"],v, order, drag_adv, df, acc_length, bank_angle,W_rem_start = W_rem, rider_data=rider_data, rho=rho, m_wheels=m_wheels, g=g
            )
        except ValueError:
            # If no feasible acceleration is found, treat v as too high
            if abs(max_v - min_v) < 0.005:
                # If the difference between max_v and min_v is very small, return the current v
                return v, "valid accel phase not found", "valid accel phase not found", "valid accel phase not found", \
                    "valid accel phase not found", "valid accel phase not found"
            max_v = v
            continue

        ss_energy, ss_powers, ss_energies, ss_total_energies = ss_func(v, peel - acc_length, switch_schedule[acc_length:], rider_data, drag_adv, order, rho, Crr, g, bike_length)
        
        errors = [W_rem_updated[rider] - ss_energy[rider] for rider in order]
        # print(f"errors: {errors}")

        if any(error < 0 for error in errors):
            max_v = v
        elif any(error < precision for error in errors):
            t_tot = tfin + (32 - acc_length) * 125 / v
            return v, t_tot, errors, slope, P_const, t_half_lap, ss_powers, ss_energies, ss_total_energies, W_rem_updated, power_profile_acc, v_sim_clean
        elif abs(max_v - min_v) < 0.005:
            # If the difference between max_v and min_v is very small, return the current v
            t_tot = tfin + (32 - acc_length) * 125 / v
            return v, t_tot, errors, slope, P_const, t_half_lap, ss_powers, ss_energies, ss_total_energies, W_rem_updated, power_profile_acc, v_sim_clean
        else:
            min_v = v

# %% [markdown]
# try calling this

# %%
#reading in data from spreadsheet
# df = pd.read_excel('2024_Performance_Physiology_Data_Anonymised_v2.xlsx')
# # Load the data
# power_curve = pd.read_excel("2024_Performance_Physiology_Data_Anonymised_v2.xlsx", sheet_name="Power Curves")
df = pd.read_excel('final_data_sheet.xlsx')

rider_data = {}
W_rem = {}
chosen_athletes = [1, 2, 3, 4]  # Example athlete numbers

name_to_number = {}
number_to_name = {}
rider_data = {}
W_rem = {}

chosen_names = df["Name"].tolist()
for i, name in enumerate(chosen_names, start=1):
    name_to_number[name] = i
    number_to_name[i] = name

for rider in chosen_athletes:
    W_prime, CP, AC, Pmax, m_rider = get_rider_info(rider, df, number_to_name)
    rider_data[rider] = {
        "W_prime": W_prime,
        "CP": CP,
        "AC": AC,
        "Pmax": Pmax,
        "m_rider": m_rider,
    }
    W_rem[rider] = W_prime

drag_adv = [1, 0.58, 0.52, 0.53]
switch_schedule = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
switch_schedule2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
switch_schedule3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
switch_schedule4 = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
order = [1, 2, 3, 4]

starttime = time.time()
v_SS, t_final, W_rem, slope, P_const, t_half_lap, ss_powers, ss_energies, ss_total_energies, W_rem_acc, power_profile_acc, v_acc = combined2(accel_phase2, race_energy2, 29, switch_schedule4, drag_adv, df, rider_data, W_rem, P0 = 50, order = order)
print(v_SS, t_final, W_rem, slope, P_const, t_half_lap)
endtime = time.time()
print(f"Time taken: {endtime - starttime:.2f} seconds")

# %%
print(f"Steady State Velocity: {v_SS:.2f} m/s")
print(f"Total time: {t_final:.2f} seconds")
print(f"Remaining W': {W_rem}")
print(f"Remaining W' after acceleration: {W_rem_acc}")
print(f"Slope: {slope:.2f} W/m")
print(f"Constant Power: {P_const:.2f} W")
print(f"Time to reach half lap: {t_half_lap:.2f} seconds")
print(f"Power profile: {ss_powers}")
print(f"Power profile for rider 1: {ss_powers[1]} and length: {len(ss_powers[1])}")
print(f"Race energy profile: {ss_energies}")
print(f"W' profile for rider 1: {ss_energies[1]} and length: {len(ss_energies[1])}")
print(f"Total energy profile: {ss_total_energies}")
print(f"Total energy profile for rider 1: {ss_total_energies[1]} and length: {len(ss_total_energies[1])}")
print(f"Power profile for acceleration: {power_profile_acc}")
print(f"Power profile for acceleration for rider 1: {power_profile_acc[1]} and length: {len(power_profile_acc[1])}")

# %%
def bar_chart(rider_data, order, W_rem, num_to_name, rider_colors ):

    for rider in order:
        # print(f"Rider: {rider}")
        percent_left = {rider: (W_rem[rider-1] / rider_data[rider]["W_prime"]) * 100 for rider in order}
        # print(f"Percent left: {percent_left}")
        percent_depleted = {rider: 100 - percent_left[rider] for rider in order}
        # print(f"Percent depleted: {percent_depleted}")

    # riders = [f"Rider {r}" for r in percent_depleted.keys()]
    riders = [num_to_name[r] for r in percent_depleted.keys()]
    values = list(percent_depleted.values())
    colors = [rider_colors[r] for r in percent_depleted.keys()]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(riders, values, color=colors)
    ax.set_ylabel("W′ depletion (%)")
    ax.set_title("Final W′ depletion by rider")
    ax.set_ylim(0, max(values) + 10)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                f"{val:.1f} %", ha="center", va="bottom", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=.4)
    fig.tight_layout()
    return fig



# %%
def plot_power_profile_over_half_laps(
    ss_powers, rider_data, order,
    P0, slope, t_half_lap, P_const,
    switch_schedule, rider_colors, num_to_name ):


    acc_half_laps = 3
    P_ramp_end = P0 + slope * t_half_lap

    switch_locations = [i for i, x in enumerate(switch_schedule) if x == 1]

    # Compute steady-state segment lengths
    f_ss = format_ss(switch_schedule[acc_half_laps:])
    # print(f"Segment lengths: {f_ss}")

    fig, ax = plt.subplots(figsize=(14, 6))

    for r in order:
        x_vals = []
        y_vals = []

        # Acceleration ramp: linear from P0 to P_ramp_end
        x_vals += [0, 1]
        y_vals += [P0, P_ramp_end]

        # Flat part of acceleration at P_const
        x_vals += [1, acc_half_laps]
        y_vals += [P_const, P_const]

        # Steady-state power segments
        hl_pos = acc_half_laps
        # for i in range(len(f_ss)):
        #     print(f"Rider {r}: len(ss_powers[r]) = {len(ss_powers[r])}, len(f_ss) = {len(f_ss)}")
        #     x_vals.append(hl_pos)
        #     x_vals.append(hl_pos + f_ss[i])
        #     y_vals.append(ss_powers[r][i])
        #     y_vals.append(ss_powers[r][i])
        #     hl_pos += f_ss[i]
        num_segments = len(ss_powers[r])
        print(f"Rider {r}: num_segments = {num_segments}")
        for i in range(num_segments):
            x_vals.append(hl_pos)
            x_vals.append(hl_pos + f_ss[i])
            y_vals.append(ss_powers[r][i])
            y_vals.append(ss_powers[r][i])
            hl_pos += f_ss[i]

            
        # ax.plot(x_vals, y_vals, label=f"Rider {r}", color=rider_colors[r])
        # ax.plot(x_vals, y_vals, label=f"Rider {r}", color=rider_colors[r], linewidth=2.5)
        ax.plot(x_vals, y_vals, color=rider_colors[r], linewidth=2.5)

        

    # Shade acceleration phase
    ax.axvspan(0, acc_half_laps, color='gray', alpha=0.3, label="Acceleration Phase")



    ax.set_xlabel("Half Laps")
    ax.set_ylabel("Power Output (W)")
    ax.set_title("Power Profile Over Half Laps")
    ax.legend(fontsize=14)
    ax.grid(False)
    return fig



# %%
def plot_power_table(
    ss_powers, order, P0, slope, t_half_lap, P_const,
    switch_schedule, rider_colors, power_profile_acc,
    W_rem_acc, rider_data, ss_wprime_profile, num_to_name ):

    acc_half_laps = 3
    P_ramp_end = P0 + slope * t_half_lap

    # Compute steady-state segment lengths
    f_ss = format_ss(switch_schedule[acc_half_laps:])
    print(f"Segment lengths: {f_ss}")

    # Compute W′ depletion during acceleration
    wprime_acc_depletion = {
        r: rider_data[r]["W_prime"] - W_rem_acc[r] for r in order
    }


    # === Filter out zero-length segments ===
    nonzero_indices = [i for i, val in enumerate(f_ss) if val > 0]

    # === Build table values ===
    power_table = {}
    max_cols = 0
    for r in order:
        acc_min = np.min(power_profile_acc[r])
        acc_max = np.max(power_profile_acc[r])
        acc_wprime = wprime_acc_depletion[r]
        acc_label = f"{acc_min:.0f}-{acc_max:.0f}W\n{acc_wprime:.0f} J"
        row = [acc_label]

        for i in nonzero_indices:
            if i < len(ss_powers[r]):
                wprime_ss = ss_wprime_profile[r][i] if i < len(ss_wprime_profile[r]) else 0
                row.append(f"{ss_powers[r][i]:.0f}W\n{wprime_ss:.0f} J")

        power_table[r] = row
        max_cols = max(max_cols, len(row))


    # Pad shorter rows
    for r in order:
        while len(power_table[r]) < max_cols:
            power_table[r].append("")

    # === Plot table ===
    n_rows = len(order) + 1
    n_cols = max_cols
    width, height = 1.0 / n_cols, 1.0 / n_rows

    fig, ax = plt.subplots(figsize=(1.5 * n_cols, 2 + len(order)))
    ax.axis("off")
    tbl = Table(ax, bbox=[0, 0, 1, 1])

    # === Header row with segment lengths ===
    for j in range(max_cols):
        if j == 0:
            header_text = f"Accel\n({acc_half_laps+0.5} half laps)"
        elif j == max_cols - 1:
            hl_idx = nonzero_indices[j - 1] if j - 1 < len(nonzero_indices) else "-"
            hl = f_ss[hl_idx] if hl_idx != "-" else "-"
            header_text = f"Seg {j}\n({hl-0.5} half laps)"
        else:
            hl_idx = nonzero_indices[j - 1] if j - 1 < len(nonzero_indices) else "-"
            hl = f_ss[hl_idx] if hl_idx != "-" else "-"
            header_text = f"Seg {j}\n({hl} half laps)"
        cell = tbl.add_cell(0, j, width, height, text=header_text, loc='center', facecolor="#f0f0f0")
        cell.get_text().set_fontsize(18)

    # === Power rows ===
    for i, r in enumerate(order):
        for j in range(n_cols):
            cell = tbl.add_cell(i + 1, j, width, height, text=power_table[r][j],
                                loc='center', facecolor=rider_colors[r])
            cell.get_text().set_fontsize(18)
        # Rider label column
        # cell = tbl.add_cell(i + 1, -1, width / 2, height, text=f"Rider {r}",
        #                     loc='right', facecolor=rider_colors[r])
        cell = tbl.add_cell(i + 1, -1, width / 2, height, text=f"{num_to_name[r]}",
                            loc='right', facecolor=rider_colors[r])
        cell.get_text().set_fontsize(18)

    ax.add_table(tbl)
    ax.set_xlabel("Half Laps")
    ax.set_ylabel("Power Output (W)")
    ax.set_title("Power Profile Over Half Laps")
    ax.legend(fontsize=14)
    ax.grid(False)
    return fig


# %%
def velocity_profile(v_acc, v_SS,  t_final, dt=0.05):
    x_vals = []
    y_vals = []

    race_end = len(v_acc)*dt

    for i in range(len(v_acc)):
        x_vals.append(i * dt)
        y_vals.append(v_acc[i])

    # Add steady-state velocity after acceleration phase
    # x_vals.append(t_half_lap)
    # y_vals.append(v_SS)
    x_vals.append(t_final)
    y_vals.append(v_SS)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_vals, y_vals, label="Velocity Profile", color= "#02534D", linewidth=2.5)
    ax.axvline(x=race_end, color='gray', linestyle='--', label="End of Acceleration Phase")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity Profile Over Time")
    ax.legend()
    fig.tight_layout()

    return fig

# %%
rider_colors = {
    1: "#C8E6C9",  
    2: "#388E3C",  
    3: "#02534D",  
    4: "#808080",  
}
def plotting(rider_data, W_rem, switch_schedule, order, v_SS, t_final, slope, P_const, t_half_lap, ss_powers, ss_energies, ss_total_energies, W_rem_acc, power_profile_acc, v_acc, rider_colors, num_to_name, P0 = 50, dt= 0.05):
    bar_chart(rider_data, order, W_rem, num_to_name , rider_colors)
    plot_power_profile_over_half_laps(
        ss_powers=ss_powers,
        rider_data=rider_data,
        order=order,
        P0=P0,
        slope=slope,
        t_half_lap=t_half_lap,
        P_const=P_const,
        switch_schedule=switch_schedule,
        rider_colors=rider_colors,
        num_to_name = num_to_name)
    plot_power_table(
        ss_powers, order, P0, slope, t_half_lap, P_const,
        switch_schedule, rider_colors, power_profile_acc,
        W_rem_acc, rider_data, ss_energies, num_to_name)
    velocity_profile(v_acc, v_SS, t_final, dt)
    

# %%
rider_colors = {
    1: "#1f77b4",  # blue
    2: "#ff7f0e",  # orange
    3: "#2ca02c",  # green
    4: "#d62728",  # red
}
plotting(rider_data, W_rem, switch_schedule4, order, v_SS, t_final, slope, P_const, t_half_lap, ss_powers, ss_energies, ss_total_energies, W_rem_acc, power_profile_acc, v_acc, rider_colors, number_to_name)

# %% [markdown]
# calling what comes out of the optimization

# %%
#reading in data from spreadsheet
df = pd.read_excel('final_data_sheet.xlsx')
# Load the data
power_curve = pd.read_excel("final_data_sheet.xlsx", sheet_name="Power Curves")

rider_data = {}
W_rem = {}
chosen_athletes = [1, 2, 3, 4]  # Example athlete numbers

name_to_number = {}
number_to_name = {}
rider_data = {}
W_rem = {}

chosen_names = df["Name"].tolist()
for i, name in enumerate(chosen_names, start=1):
    name_to_number[name] = i
    number_to_name[i] = name

for rider in chosen_athletes:
    W_prime, CP, AC, Pmax, m_rider = get_rider_info(rider, df, number_to_name)
    rider_data[rider] = {
        "W_prime": W_prime,
        "CP": CP,
        "AC": AC,
        "Pmax": Pmax,
        "m_rider": m_rider,
    }
    W_rem[rider] = W_prime

drag_adv = [1, 0.58, 0.52, 0.53]

schedule = [5, 10, 18, 24, 32]
full_switch_schedule = [0] * 32
for point in schedule:
    if 0 <= point < 32:
        full_switch_schedule[int(point)] = 1
# switch_schedule = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
order = [3, 4, 1, 2]
peel_location = 24

starttime = time.time()
v_SS, t_final, W_rem, slope, P_const, t_half_lap, ss_powers, ss_energies, ss_total_energies, W_rem_acc, power_profile_acc, v_acc = combined2(accel_phase2, race_energy2, peel_location, full_switch_schedule, drag_adv, df, rider_data, W_rem, P0 = 50, order = order)
print(v_SS, t_final, W_rem, slope, P_const, t_half_lap)
endtime = time.time()
print(f"Time taken: {endtime - starttime:.2f} seconds")


# %%
# rider_colors = {
#     1: "#1f77b4",  # blue
#     2: "#ff7f0e",  # orange
#     3: "#2ca02c",  # green
#     4: "#d62728",  # red
# }
# plotting(rider_data, W_rem, switch_schedule, order, v_SS, t_final, slope, P_const, t_half_lap, ss_powers, ss_energies, ss_total_energies, W_rem_acc, power_profile_acc, v_acc, rider_colors, number_to_name)



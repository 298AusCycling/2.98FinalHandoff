# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import math
import random
from scipy.optimize import root_scalar
from scipy.stats import truncnorm
import itertools
from itertools import permutations, combinations
import traceback, logging
logger = logging.getLogger(__name__)

# %% [markdown]
# putting in the functions we need

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

def combined(acc_func, ss_func, peel, switch_schedule, drag_adv, df, rider_data, W_rem,
             order=[1,2,3,4], 
             min_v=15, max_v=22, precision=200, acc_length=3, 
             rho=1.225, Crr=0.0018, g=9.80665, bike_length=2.1, 
             m_wheels=0.75, v0=1.5, P0=500, bank_angle=np.radians(12)):

    leader = order[0]

    while True:
        v = (min_v + max_v) / 2
        # print(f"Trying v: {v}")
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

# %%
### Black box model that is a stand in for our actual function 
# comment out time.sleep(0.25) to make it instantaneous
# or, change to the number of seconds we want this to take...

# question: do we also need to include acceleration_length (number of half laps)?
counter = 0 
def black_box(schedule, peel, initial_order, acceleration_length, drag_adv, df, rider_data, W_rem, P0=50):
    try:
        # Create a full 32-length switch schedule from switch point list
        full_switch_schedule = [0] * 32
        for point in schedule:
            if 0 <= point < 32:
                full_switch_schedule[int(point)] = 1

        v_out, t_out, *_ = combined(
            accel_phase,
            race_energy,
            peel,
            full_switch_schedule,
            drag_adv,
            df,
            rider_data,
            W_rem,
            order=initial_order,
            acc_length=acceleration_length,
            P0=P0
        )

        global counter
        counter += 1
        return t_out  # total race time
    except Exception:
        logger.exception("black_box crashed — dumping context")



# %%
### this is helpful in the genetic algorithm (allows for the function to be minimum of this, but not maximum) 

def sample_truncated_normal(center, min_, max_, std=1):
    a = (min_ - 1 - center) / std
    b = (max_ - 1 - center) / std
    return round(truncnorm.rvs(a, b, loc=center, scale=std))


# uncomment below to see how it works (the max would be considered too big): 
#list_stn = []
#for i in range(1000): 
#    list_stn.append(sample_truncated_normal(center=6, min_=4, max_=8))

#bins = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
#plt.hist(list_stn, bins=bins, edgecolor='black', align='mid')
#plt.xticks([3, 4, 5, 6, 7, 8, 9])
#plt.title('Histogram')
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.show()

# %%
# this takes a list, and replaces the closest element to "peel" with "peel" 

def replace_with_peel(peel, relevant_list):
    closest_index = min(range(len(relevant_list)), key=lambda i: abs(relevant_list[i] - peel))
    relevant_list[closest_index] = peel
    #print("did it")
    #print(relevant_list)
    return relevant_list

# %%
# function that creates your list of 10 (11 if you count the parent) jittered options 

def create_jittered_kids(parent, acceleration_length, num_changes, num_children, peel, parent_list):
    parent_list.append(parent) #first append the parent to parent_list 
    warm_start = parent #for help with naming 
    #print("initial:", warm_start)
    all_children = []
    #all_children.append(warm_start) ### CHANGE HERE
    all_children.append(replace_with_peel(peel, warm_start[:]))
    #num_children=10000
    j = 1
    while j <= num_children: 
        #print("option number:", j)
        jittered_list = []
        for i in range(num_changes):
            if i == 0: 
                prev_jitter = 3
                while prev_jitter <= acceleration_length:
                    prev_jitter = sample_truncated_normal(center=warm_start[i], min_=warm_start[i]-(warm_start[i+1]-warm_start[i]), max_=warm_start[i+1])
                    #print("tried this: ", jitter)
                #print("first_jitter:", prev_jitter) 
                jittered_list.append(prev_jitter)
                ### CODE for what to do for the first one 
            elif i != 0 and i != num_changes-1:
                #print("initial:", warm_start[i])
                jitter = -1
                while jitter <= prev_jitter:
                    jitter = sample_truncated_normal(center=warm_start[i], min_=warm_start[i-1], max_=warm_start[i+1])
                #print("jitter:", jitter)
                jittered_list.append(jitter)
                prev_jitter = jitter 
                # CODE FOR FIXING 
            else: 
                # I don't think we need all this, but whatever 
                last_jitter = 50
                while last_jitter > 32 or last_jitter <= prev_jitter:
                    last_jitter = sample_truncated_normal(center=warm_start[i], min_=warm_start[i-1], max_=warm_start[i]+warm_start[i]-warm_start[i-1])
                    #last_jitter = sample_truncated_normal(center=36, min_=32, max_=40)
                #print("last_jitter:",last_jitter) 
                jittered_list.append(last_jitter)
        if jittered_list not in all_children: 
            all_children.append(replace_with_peel(peel, jittered_list[:])) #CHANGE HERE, ADDED THIS
            j = j+1 # and this! 
            #if peel in jittered_list: ### IF WE ITERATE THROUGH PEELS, REMOVE THIS IF STATEMENT (OR MAKE IT ARBITRARY) 
                #all_children.append(jittered_list)
                #j = j+1
                


    return all_children, parent_list
    

# %%
# returns teh top 4 from a list (can be changed to not be 4) 

# def best_from_list(children, tested_list,peel,initial_order):
#     my_dict = {}
#     for child in children: 
#         if child not in tested_list: # ADD A FOR LOOP HERE, FOR PEEL IN CHILDREN: 
#             the_time = black_box(child, peel, initial_order)
#             tested_list.append(child)
#             if len(my_dict) < 4 or the_time < max(my_dict.values()):
#                 my_dict[tuple(child)] = the_time
#                 if len(my_dict) > 4:
#                     # Remove the key with the largest value
#                     del my_dict[max(my_dict, key=my_dict.get)]
#     return my_dict, tested_list

def best_from_list(children, tested_list, peel, initial_order, acceleration_length, drag_adv, df, rider_data, W_rem, num_seeds, P0=50):
    my_dict = {}
    for child in children:
        if child not in tested_list:
            the_time = black_box(child, peel, initial_order, acceleration_length, drag_adv, df, rider_data, W_rem, P0)
            tested_list.append(child)
            if len(my_dict) < num_seeds or the_time < max(my_dict.values()):
                my_dict[tuple(child)] = the_time
                if len(my_dict) > num_seeds:
                    del my_dict[max(my_dict, key=my_dict.get)]
    return my_dict, tested_list


# %%
# # make a function that, given a peel (half lap number), initial_order, and num_changes, outputs the fastest (according to genetic algorithm) 
# # maybe also have inputs for number of jittered children from seed, number of seeds chosen at each round, and number of rounds 

# def genetic_algorithm(peel, initial_order, acceleration_length, num_changes, num_children = 10, num_seeds = 4, num_rounds = 5):
#     warm_start = np.linspace(acceleration_length+1, 31.9, num=num_changes, dtype=int).tolist()
#     parent_list = []
#     fxn_output = create_jittered_kids(warm_start, acceleration_length,num_changes,num_children,peel,parent_list)
#     all_children_from_fxn = fxn_output[0]
#     parent_list = fxn_output[1]
    

#     # now, we have a function that takes an input, and gives 10 jittered lists. (along with keeping up the parent_list
#     # next, we need to take in that list of lists, and output the 4 best, along with keeping a running tab on what we have already tried

#     tested_list = []
#     fxn_output2 = best_from_list(all_children_from_fxn, tested_list,peel,initial_order)
#     dict_of_top_4 = fxn_output2[0]
#     tested_list = fxn_output2[1]

#     # THAT WAS INITIAL ROUND, WE DON'T REALLY COUNT IT 

#     # now, we have a dict, where the keys are the top 4 (we need to re-listify them) 
#     list_of_active_parents = [list(key) for key in dict_of_top_4.keys()]
    
#     for i in range(num_rounds):
#         for a_list in list_of_active_parents:
#             if a_list not in parent_list: 
#                 output = create_jittered_kids(a_list, acceleration_length,num_changes,num_children,peel,parent_list)
#                 all_kids = output[0]
#                 parent_list = output[1]
#                 for a_kid in all_kids: 
#                     if a_kid not in tested_list: # ADD A FOR LOOP HERE, FOR PEEL IN A_KID: 
#                         time_for_this_kid = black_box(a_kid, peel, initial_order)
#                         tested_list.append(a_kid)
#                         if time_for_this_kid < max(dict_of_top_4.values()):
#                             dict_of_top_4[tuple(a_kid)] = time_for_this_kid
#                             del dict_of_top_4[max(dict_of_top_4, key=dict_of_top_4.get)]
#         list_of_active_parents = [list(key) for key in dict_of_top_4.keys()]
        

#     # AT THIS POINT, we should have dict_of_top_4 that has been updated quite well...
#     # IMPROVEMENT: KEEP A THRESHOLD FOR IMPROVEMENT THAT ONCE WE MEET IT, WE ARE GOOD
#     # OR JUST KEEP GOING AFTER NUM_ROUNDS IF WE ARE IMPROVING 

#     time_of_race = min(dict_of_top_4.values())
#     schedule_of_switches = min(dict_of_top_4, key=dict_of_top_4.get)

#     # Or, we could just have it output all 4 

    
#         # for list in list_of_active_parents 
#         # if not in parent_list:
#             # create_jittered_kids
#             # update parent list
#             # for loop of children from jittering
#                 # best_from_list added to dict_of_top_4 if they are good enough 
#                 # also update list_of_active_parents ??? NO!
#         # after we have gone through entire list_of_active_parents, create list_of_active_parents_2
#         #list_of_active_parents = list_of_active_parents2
#             # or, do we break if these are the same? I vote no, we want to keep jittering and trying again 

#     # ENSURE WE DO THIS EACH TIME WE ARE CALLING A FUNCTION!!!
#     # maintain a master list so we know what we have already tried 
#     # maintain a parent list so we know what we have already jittered 
   
#     # also could do more than 1... 

#     sorted_dict = dict(sorted(dict_of_top_4.items(), key=lambda item: item[1]))

    
#     return time_of_race, schedule_of_switches, sorted_dict
    

def genetic_algorithm(peel, initial_order, acceleration_length, num_changes,
                      drag_adv, df, rider_data, W_rem,
                      num_children=10, num_seeds=4, num_rounds=5, P0=50):
    
    warm_start = np.linspace(acceleration_length+1, 31.9, num=num_changes, dtype=int).tolist()
    parent_list = []

    fxn_output = create_jittered_kids(warm_start, acceleration_length, num_changes, num_children, peel, parent_list)
    all_children_from_fxn = fxn_output[0]
    parent_list = fxn_output[1]

    tested_list = []
    dict_of_top_4, tested_list = best_from_list(
        all_children_from_fxn,
        tested_list,
        peel,
        initial_order,
        acceleration_length,
        drag_adv,
        df,
        rider_data,
        W_rem,
        num_seeds,
        P0
    )

    list_of_active_parents = [list(key) for key in dict_of_top_4.keys()]
    
    for i in range(num_rounds):
        for a_list in list_of_active_parents:
            if a_list not in parent_list:
                all_kids, parent_list = create_jittered_kids(a_list, acceleration_length, num_changes, num_children, peel, parent_list)
                for a_kid in all_kids:
                    if a_kid not in tested_list:
                        time_for_this_kid = black_box(a_kid, peel, initial_order, acceleration_length, drag_adv, df, rider_data, W_rem, P0)
                        tested_list.append(a_kid)
                        if time_for_this_kid < max(dict_of_top_4.values()):
                            dict_of_top_4[tuple(a_kid)] = time_for_this_kid
                            del dict_of_top_4[max(dict_of_top_4, key=dict_of_top_4.get)]
        list_of_active_parents = [list(key) for key in dict_of_top_4.keys()]

    time_of_race = min(dict_of_top_4.values())
    schedule_of_switches = min(dict_of_top_4, key=dict_of_top_4.get)
    sorted_dict = dict(sorted(dict_of_top_4.items(), key=lambda item: item[1]))

    return time_of_race, schedule_of_switches, sorted_dict


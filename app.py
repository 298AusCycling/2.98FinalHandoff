import streamlit as st
import pandas as pd
import time
import sqlite3
import json
from datetime import datetime
from final_plots import combined2, get_rider_info, accel_phase2, race_energy2, bar_chart, plot_power_table, plot_power_profile_over_half_laps, velocity_profile
import matplotlib
matplotlib.use("Agg")
import requests
import io, base64

st.set_page_config(layout="wide")
main_title = st.title("Team Pursuit Race Simulator")
if "opt_job_id" not in st.session_state:
    st.session_state.opt_job_id = None
if "opt_polling" not in st.session_state:
    st.session_state.opt_polling = False

# --- Setup database ---
def fig_to_png_bytes(fig) -> bytes:
    """Return a Matplotlib figure as raw PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)           # free VRAM
    return buf.getvalue()

def png_bytes_to_base64(b: bytes) -> str:
    """Encode raw PNG bytes for storage / JSON."""
    return base64.b64encode(b).decode("ascii")

conn = sqlite3.connect("simulations.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS optimizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    total_races INTEGER,
    runtime_seconds REAL,
    result_json TEXT
)
""")
conn.commit()

cursor.execute("""
CREATE TABLE IF NOT EXISTS simulations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    chosen_athletes TEXT,
    start_order TEXT,
    switch_schedule TEXT,
    peel_location INTEGER,
    final_time REAL,
    final_distance REAL,
    final_half_lap_count INTEGER,
    W_rem TEXT
)
""")
conn.commit()

drafting_percents = [1.0, 0.58, 0.52, 0.53]

# --- Helper functions ---
def switch_schedule_description(switch_schedule):
    return [i + 1 for i, v in enumerate(switch_schedule) if v == 1]

def save_simulation_to_db(record):
    cursor.execute("""
        INSERT INTO simulations (
            timestamp, chosen_athletes, start_order, switch_schedule,
            peel_location, final_time, final_distance,
            final_half_lap_count, W_rem,
            fig1_png, fig2_png, fig3_png, fig4_png        -- NEW
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.fromtimestamp(record["timestamp"]).isoformat(),
        json.dumps(record["chosen_athletes"]),
        json.dumps(record["start_order"]),
        json.dumps(record["switch_schedule"]),
        record["peel_location"],
        record["final_time"],
        record["final_distance"],
        record["final_half_lap_count"],
        json.dumps(record["W_rem"]),
        record["fig1_png"], record["fig2_png"],
        record["fig3_png"], record["fig4_png"],         # NEW
    ))
    conn.commit()

def save_optimization_to_db(runtime, total_races, top_results):
    cursor.execute("""
        INSERT INTO optimizations (
            timestamp, total_races, runtime_seconds, result_json
        ) VALUES (?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        total_races,
        runtime,
        json.dumps(top_results),
    ))
    conn.commit()

def plot_switch_strategy(start_order, switch_schedule):
    import matplotlib.pyplot as plt

    colors = {rider: color for rider, color in zip(start_order, ['#C8E6C9', '#388E3C', '#02534D', '#808080'])}
    lead_segments = []
    leader_index = 0
    start = 0

    for i, switch in enumerate(switch_schedule):
        if switch == 1:
            duration = i + 1 - start
            lead_segments.append({"rider": start_order[leader_index % len(start_order)], "start": start, "duration": duration})
            start = i + 1
            leader_index += 1

    if start < len(switch_schedule):
        lead_segments.append({"rider": start_order[leader_index % len(start_order)], "start": start, "duration": len(switch_schedule) - start + 1})

    # Group segments by rider
    segments_by_rider = {r: [] for r in start_order}
    for seg in lead_segments:
        segments_by_rider[seg["rider"]].append(seg)

    fig, ax = plt.subplots(figsize=(12, 4))
    y_levels = {rider: i for i, rider in enumerate(reversed(start_order))}

    for rider in start_order:
        y = y_levels[rider]
        rider_segs = sorted(segments_by_rider[rider], key=lambda x: x["start"])
        prev_end = 0
        for seg in rider_segs:
            x = seg["start"]
            w = seg["duration"]
            ax.broken_barh([(x, w)], (y - 0.4, 0.8), facecolors=colors[rider])
            ax.text(x + w / 2, y, f'{w}', ha="center", va="center", fontsize=9, color="white")

            if prev_end < x:
                rest_len = x - prev_end
                mid = (prev_end + x) / 2
                ax.text(mid, y + 0.25, f'{rest_len}', ha="center", va="bottom", fontsize=8, color="black")
            prev_end = x + w

    ax.set_yticks(list(y_levels.values()))
    ax.set_yticklabels(r for r in reversed(start_order))
    ax.set_xlabel("Half-laps")
    ax.set_ylabel("Rider")
    ax.set_title("Turn Strategy")
    ax.set_xlim(0, len(switch_schedule) + 2)
    ax.grid(True, axis="x")
    st.pyplot(fig)
    plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

model_type = st.radio("Select Model Type", ["Optimization", "Coach Input"], index=None)

if model_type == "Coach Input":
    st.markdown('***User Input Model***')
    # --- Main Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["Data Input", "Advanced Settings", "Simulate Race", "Previous Simulations"])

    # --- Tab 1: Upload Data ---
    with tab1:
        uploaded_file = st.file_uploader("Upload Performance Data Excel File", type=["xlsx"])

    # --- Tab 2: Advanced Settings ---
    with tab2:
        rho_input = st.number_input("**Air Density (kg/m³)**", value=1.225, step=0.001, format="%.3f")
        Crr_input = st.number_input("**Rolling Resistance (Crr)**", value=0.0018, step=0.0001, format="%.4f")
        v0_input = st.number_input("**Initial Velocity (m/s)**", value=0.5, step=0.01, format="%.2f")
        default_values = [1.0, 0.58, 0.52, 0.53]
        drag_adv = []
        for i in range(4):
            value = st.number_input(
                f"Drag advantage for Rider {i + 1}",
                min_value=0.0,
                max_value=1.5,
                step=0.01,
                value=default_values[i]
            )
            drag_adv.append(value)
        p0_input = st.number_input("**Initial Power**", value = 50, step = 1)

        

    # --- Tab 3: Simulate Race ---
    with tab3:
        if uploaded_file:
            left_col, right_col = st.columns([1, 3])

            with left_col:
                df_athletes = pd.read_excel(uploaded_file, engine="openpyxl")

                available_athletes = df_athletes["Name"].to_list()

                chosen_athletes = st.multiselect("Select 4 Athletes", available_athletes)
                st.markdown(f"Selected Riders: {sorted(chosen_athletes)}.")

                if len(chosen_athletes) == 4:
                    start_order = st.multiselect("Initial Rider Order", sorted(chosen_athletes))
                    st.markdown(f"Initial Starting Order: {start_order}")

                    st.subheader("Turn Schedule (32 half-laps)")
                    switch_schedule = []
                    peel_schedule = []

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Turn (1 = Turn after this half-lap)**")
                        for i in range(32):
                            val = st.checkbox(f"{i+1}", key=f"switch_{i}")
                            switch_schedule.append(1 if val else 0)

                    with col2:
                        st.markdown("**Peel (1 = 3rd rider peel here)**")
                        for i in range(32):
                            val = st.checkbox(f"{i+1}", key=f"peel_{i}")
                            peel_schedule.append(1 if val else 0)

                    try:
                        peel_location = peel_schedule.index(1)
                    except ValueError:
                        peel_location = None

                    simulate = st.button("Simulate Race")
                    if simulate:
                        st.success("Simulation Complete!")
                else:
                    simulate = False
                    st.warning("Please select exactly 4 riders.")

            with right_col:
                if simulate and start_order and peel_location is not None:
                    with st.spinner("Running simulation..."):
                        # Load data from uploaded file
                        df_athletes = pd.read_excel(uploaded_file, engine="openpyxl")
                        chosen_athletes_names = chosen_athletes                

                        name_to_number = {name: i for i, name in enumerate(chosen_athletes_names, 1)}
                        number_to_name = {i: name for name, i in name_to_number.items()}

                        # numeric IDs for the rest of the code
                        chosen_athletes_nums = list(number_to_name.keys())     # [1, 2, 3, 4]
                        start_order_nums     = [name_to_number[n] for n in start_order]

                        # build rider_data and initial W′
                        rider_data = {}
                        W_rem      = {}
                        for r in chosen_athletes_nums:                         # r = 1,2,3,4
                            Wp, CP, AC, Pmax, m = get_rider_info(r, df_athletes, number_to_name)
                            rider_data[r] = {"W_prime": Wp, "CP": CP, "AC": AC, "Pmax": Pmax, "m_rider": m}
                            W_rem[r]      = Wp

                        # run the simulation
                        v_SS, t_final, W_rem, slope, P_const, t_half_lap, \
                        ss_powers, ss_energies, ss_total_energies, \
                        W_rem_acc, power_profile_acc, v_acc = combined2(
                                accel_phase2, race_energy2, peel_location,
                                switch_schedule, drag_adv,
                                df_athletes, rider_data, W_rem,
                                P0=50, order=start_order_nums
                        )
                            
                    with st.container():
                        row1 = st.columns(3)
                        num_to_name = {}
                        for i, name in enumerate(chosen_athletes, start=1):
                            name_to_number[name] = i
                            num_to_name[i] = name
                        with row1[0]:
                            st.markdown("**Total Time**")
                            st.markdown(f"{t_final:.2f} s")
                        with row1[1]:
                            st.markdown("**Target Velocity**")
                            st.markdown(f"{v_SS:.2f} m/s  \n({v_SS*3.6:.1f} km/h)")

                        with row1[2]:
                            st.markdown("**Turns:**")
                            switches = switch_schedule_description(switch_schedule)
                            st.markdown(", ".join(str(s) for s in switches))

                        st.subheader("Turn Strategy Timeline")
                        plot_switch_strategy(start_order, switch_schedule)
                        rider_colors = {
                            1: "#C8E6C9",  
                            2: "#388E3C",  
                            3: "#02534D",  
                            4: "#808080",  
                        }
                        st.subheader("Plots")
                        fig1 = bar_chart(rider_data, start_order_nums, W_rem, number_to_name, rider_colors)
                        st.pyplot(fig1)              
                        fig1_png = fig_to_png_bytes(fig1)  

                        fig2 = plot_power_table(
                            ss_powers, start_order_nums, 50, slope, t_half_lap, P_const,
                            switch_schedule, rider_colors, power_profile_acc,
                            W_rem_acc, rider_data, ss_energies, num_to_name
                        )
                        st.pyplot(fig2)              
                        fig2_png = fig_to_png_bytes(fig2)   

                        fig3 = plot_power_profile_over_half_laps(
                            ss_powers, rider_data, start_order_nums, 50, slope, t_half_lap, P_const,
                            switch_schedule, rider_colors, v_SS
                        )
                        st.pyplot(fig3)              
                        fig3_png = fig_to_png_bytes(fig3)   

                        fig4 = velocity_profile(v_acc, v_SS, t_final, dt=0.05)
                        st.pyplot(fig4)              
                        fig4_png = fig_to_png_bytes(fig4) 
                        
                    st.subheader("W′ Remaining per Rider:")
                    for name in start_order:            
                        idx = name_to_number[name]       # 1–4
                        st.write(f"**{name}**: {W_rem[idx-1]:.1f} J")

                        simulation_record = {
                            "timestamp": time.time(),
                            "chosen_athletes": chosen_athletes,
                            "start_order": start_order,
                            "switch_schedule": switch_schedule,
                            "peel_location": peel_location,
                            "final_time": t_final,
                            "final_distance": None,
                            "final_half_lap_count": None,
                            "W_rem": W_rem,
                        "fig1_png": fig1_png,
                        "fig2_png": fig2_png,
                        "fig3_png": fig3_png,
                        "fig4_png": fig4_png,
                    }
                    save_simulation_to_db(simulation_record)

    # --- Tab 4: Previous Simulations ---
    with tab4:
        st.subheader("Download Past Simulations")
        cursor.execute("SELECT * FROM simulations ORDER BY id DESC")
        all_rows = cursor.fetchall()

        if all_rows:
            df_download = pd.DataFrame([
            {
                "id": row[0],
                "timestamp": row[1],
                "chosen_athletes": json.loads(row[2]),
                "start_order": json.loads(row[3]),
                "switch_schedule": json.loads(row[4]),
                "peel_location": row[5],
                "final_time": row[7],
                "final_distance": row[8],
                "final_half_lap_count": row[9],
                "W_rem": json.loads(row[10]),
                "fig1_png": row[11],          # NEW
                "fig2_png": row[12],          # NEW
                "fig3_png": row[13],          # NEW
                "fig4_png": row[14],          # NEW
            }
            for row in all_rows
        ])

            st.download_button(
                label="Download Simulations as CSV",
                data=df_download.to_csv(index=False).encode("utf-8"),
                file_name="simulations.csv",
                mime="text/csv",
            )

            for i, row in df_download.iterrows():
                with st.expander(f"Simulation #{row['id']} — {row['timestamp']}"):
                    st.write(f"**Chosen Athletes:** {row['chosen_athletes']}")
                    st.write(f"**Start Order:** {row['start_order']}")
                    # st.write(f"**Final Order:** {row['final_order']}")
                    st.write(f"**Peel Location:** {row['peel_location']}")
                    st.write(f"**Total Time:** {row['final_time']:.2f} seconds")
                    st.write(f"**Turn Schedule:** {switch_schedule_description(row['switch_schedule'])}")
                    st.subheader("W′ Remaining per Rider:")
                    for idx, energy_left in enumerate(row["W_rem"]):
                        st.write(f"**Rider {idx+1}**: {float(energy_left):.1f} J")
                    st.subheader("Turn Strategy Timeline")
                    try:
                        plot_switch_strategy(row["start_order"], row["switch_schedule"])
                    except Exception as e:
                        st.warning("Couldn't render strategy timeline for this entry.")
                    delete = st.button(f"Delete Simulation #{row['id']}", key=f"delete_{row['id']}")
                    img1 = row.get("fig1_png")
                    if img1:
                        st.image(img1, caption="Remaining W′ bar")
                    
                    img2 = row.get("fig2_png")
                    if img2:
                        st.image(img2)

                    img3 = row.get("fig3_png")
                    if img3:
                        st.image(img3)
                    
                    img4 = row.get("fig4_png")
                    if img4:
                        st.image(img4)

                    if delete:
                        cursor.execute("DELETE FROM simulations WHERE id = ?", (row["id"],))
                        conn.commit()
                        st.success(f"Simulation #{row['id']} deleted successfully.")
                        st.rerun()
        else:
            st.info("No simulations available yet.")
elif model_type == "Optimization":
    st.markdown('***Optimization Model***')
    tab5, tab6, tab7, tab8 = st.tabs(["Data Input", "Advanced Settings", "Simulate Race", "Previous Simulations"])
    with tab5: 
        uploaded_file_opt = st.file_uploader(
        "Upload Performance Data Excel File",
        type=["xlsx"],
        key="optimizer_upload",
    )

    if uploaded_file_opt:
        df_opt = pd.read_excel(uploaded_file_opt)

        # Extract numeric rider IDs, eg “M123” → 123
        available_riders = (
            df_opt["Name"].str.extract(r"M(\d+)")[0]
            .dropna()
            .astype(int)
            .tolist()
        )

        # cache for next tabs
        st.session_state["df_opt"] = df_opt
        st.session_state["available_riders"] = available_riders

        st.success(f"Loaded {len(df_opt)} rows. "
                   f"Found riders: {sorted(available_riders)}")
    else:
        st.session_state.pop("df_opt",  None)
        st.session_state.pop("available_riders", None)

    with tab6:
        rho_input_opt = st.number_input("**Air Density (kg/m³)**", value=1.225, step=0.001, format="%.3f")
        Crr_input_opt = st.number_input("**Rolling Resistance (Crr)**", value=0.0018, step=0.0001, format="%.4f")
        v0_input_opt = st.number_input("**Initial Velocity (m/s)**", value=0.5, step=0.01, format="%.2f")
    with tab7:
        if uploaded_file_opt:
            if "df_opt" not in st.session_state:
                st.info("Upload a data sheet in the *Data Input* tab first.")
                st.stop()
            df_opt          = st.session_state["df_opt"]
            available       = st.session_state["available_riders"]
            chosen_riders = st.multiselect(
                "Select exactly 4 riders for optimisation",
                options=sorted(available),
                key="chosen_riders_opt",
            )
            run_disabled = len(chosen_riders) != 4
            run_btn      = st.button("Run Optimization Model",
                                    disabled=run_disabled)
            if run_btn:
                payload = {
                    "workbook": df_opt.to_json(orient="split"),
                    "rider_ids": chosen_riders,
                    "drag_adv": [1.0, 0.58, 0.52, 0.53],
                    "rho": rho_input_opt,
                    "Crr": Crr_input_opt,
                    "v0": v0_input_opt,
                }

            
            if run_btn and not st.session_state.opt_polling:
                with st.spinner("Starting optimisation VM…"):
                    try:
                        cloud_function_url = (
                            "https://us-central1-team-pursuit-optimizer.cloudfunctions.net/start-vm-lite"
                        )
                        requests.post(cloud_function_url, timeout=60)
                    except Exception as e:
                        st.warning(f"VM start request failed (proceeding anyway): {e}")

                with st.spinner("Submitting optimisation job…"):
                    with st.spinner("Submitting optimisation job…"):
                        try:
                            r = requests.post(
                                "http://35.209.48.32:8000/run_optimization",
                                json=payload,
                                timeout=60,
                            )
                            r.raise_for_status()       
                        except requests.HTTPError as e:
                            st.error(f"HTTP {e.response.status_code}: {e.response.text}")  # ★
                            st.stop()
                        st.session_state.opt_job_id = r.json()["job_id"]
                        st.session_state.opt_polling = True
                        st.success(f"Job queued: `{st.session_state.opt_job_id}`")
                        st.rerun()          # kick off the polling loop immediately
                    # except Exception as e:
                    #     st.error(f"Could not start optimisation: {e}")

            if st.session_state.opt_polling and st.session_state.opt_job_id:
                job_id = st.session_state.opt_job_id
                status_box = st.empty()
                progress = st.progress(0)

                try:
                    resp = requests.get(f"http://35.209.48.32:8000/run_optimization/{job_id}", timeout=10)
                    data = resp.json()

                    if data.get("state") == "running":
                        pct = data.get("progress", 0)
                        progress.progress(pct, text=f"{pct}% complete")
                        status_box.info(f"Job `{job_id}` is running…")
                        time.sleep(5)
                        st.rerun()   # refresh the page and poll again

                    elif data.get("state") == "done":
                        progress.progress(100, text="Finished")
                        st.session_state.opt_polling = False

                        # Save to DB
                        save_optimization_to_db(
                            data["runtime_seconds"],
                            data["total_races_simulated"],
                            data["top_results"],
                        )

                        st.success(
                            f"Optimisation finished in {data['runtime_seconds']:.1f} s "
                            f"after {data['total_races_simulated']:,} races."
                        )

                        st.subheader("Top 5 Results")
                        for i, res in enumerate(data["top_results"], 1):
                            switches_raw = res["switches"]
                            if isinstance(switches_raw, (list, tuple)):
                                switches = ", ".join(map(str, switches_raw))
                            else:
                                switches = str(switches_raw)          
                            init_ord = "-".join(map(str, res["initial_order"]))
                            peel_at  = res["peel"]

                            st.markdown(
                                f"**#{i}** — **{res['time']:.2f} s**  \n"
                                f"• Initial order: `{init_ord}`  \n"
                                f"• Peel after half-lap: **{peel_at}**  \n"
                                f"• Switch schedule: `{switches}`"
                            )

                    elif data.get("state") == "error":
                        st.session_state.opt_polling = False
                        progress.empty()
                        st.error(f"Job failed: {data['error']}")

                    else:
                        st.session_state.opt_polling = False
                        progress.empty()
                        st.error("Unknown job status.")

                except Exception as e:
                   st.session_state.opt_polling = False
                   progress.empty()            # clear bar here too
                   st.error(f"Error contacting backend: {e}")

        else:
            st.info("Please upload a dataset first.")
    with tab8:
        st.subheader("Previous Optimization Runs")
        cursor.execute("SELECT * FROM optimizations ORDER BY id DESC")
        rows = cursor.fetchall()

        if rows:
            df_opt = pd.DataFrame([
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "total_races": row[2],
                    "runtime_seconds": row[3],
                    "top_results": json.loads(row[4])
                }
                for row in rows
            ])

            st.download_button(
                "Download as CSV",
                data=df_opt.to_csv(index=False).encode("utf-8"),
                file_name="optimizations.csv",
                mime="text/csv",
            )

            for i, row in df_opt.iterrows():
                with st.expander(f"Optimization #{row['id']} — {row['timestamp']}"):
                    for j, res in enumerate(row["top_results"], 1):
                        switches_raw = res["switches"]
                        if isinstance(switches_raw, (list, tuple)):
                            switches = ", ".join(map(str, switches_raw))
                        else:
                            switches = str(switches_raw)          # single value → just show it
                        init_ord = "-".join(map(str, res["initial_order"]))
                        peel_at  = res["peel"]

                        st.markdown(
                            f"**#{j}** — **{res['time']:.2f} s**  \n"
                            f"• Initial order: `{init_ord}`  \n"
                            f"• Peel after half-lap: **{peel_at}**  \n"
                            f"• Switch schedule: `{switches}`"
                        )
                    delete = st.button(f"Delete Simulation #{row['id']}", key=f"delete_{row['id']}")
                    if delete:
                        cursor.execute("DELETE FROM optimizations WHERE id = ?", (row["id"],))
                        conn.commit()
                        st.success(f"Simulation #{row['id']} deleted successfully.")
                        st.rerun()
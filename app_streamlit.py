import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # <-- Added for legend
import random
import time
import math
import os
import io
from shapely.geometry import Point
from fpdf import FPDF  # pip install fpdf
from PIL import Image

##############################################
# File Paths
##############################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

hospital_data_path = os.path.join(BASE_DIR, "FINAL_LIST_ALL_MASTER_NewYorkComplete.csv")
county_shapefile_path = os.path.join(BASE_DIR, "NYS_Civil_Boundaries", "Counties.shp")

# Optional logo path
logo_path = os.path.join(BASE_DIR, "NewYorkAgent.png")  # Place a PNG in the same folder (optional)

##############################################
# Load Data (Cached)
##############################################
@st.cache_data
def load_data():
    # 1) Load hospital data
    hospital_data = pd.read_csv(hospital_data_path)
    hospital_data = hospital_data[hospital_data['State'] == 'NY'].copy()

    # -----------------------------------------------------------------
    # Apply filters:
    #  - Keep only rows where 'Emergency Services' == 'Yes'
    #  - Remove rows where 'Federal' == True
    # -----------------------------------------------------------------
    # If 'Emergency Services' is always a string: 'Yes' / 'No'
    hospital_data = hospital_data[hospital_data['Emergency Services'] == 'Yes']

    # If 'Federal' is a boolean column (True/False)
    hospital_data = hospital_data[hospital_data['Federal'] != True]

    # If 'Federal' were a string ('TRUE'/'FALSE'),
    # you could do: hospital_data = hospital_data[hospital_data['Federal'].str.upper() != 'TRUE']

    # Ensure there's a "County" column that matches with shapefile's county name
    hospital_data['County'] = hospital_data['County'].str.title().str.strip()

    # Convert bed columns to numeric
    hospital_data['TOTAL BEDS, POS'] = pd.to_numeric(
        hospital_data['TOTAL BEDS, POS'], errors='coerce'
    ).fillna(0)
    hospital_data['Available Beds'] = hospital_data['TOTAL BEDS, POS']
    hospital_data['Occupancy'] = 0.0

    # 2) Load county shapefile
    county_gdf = gpd.read_file(county_shapefile_path).to_crs(epsg=4326)

    # Make sure county names match the "County" field in hospital data
    county_gdf['Name'] = county_gdf['NAME'].str.title().str.strip()

    # Use the 'POP2020' field for the county population
    county_gdf['Total Population'] = (
        pd.to_numeric(county_gdf['POP2020'], errors='coerce')
        .fillna(0)
        .astype(int)
    )

    # 3) Merge hospital data with county geometry by name only
    merged_df = hospital_data.merge(
        county_gdf[['Name', 'geometry']],
        left_on='County',
        right_on='Name',
        how='left'
    )

    # Convert to GeoDataFrame
    hospital_gdf = gpd.GeoDataFrame(
        merged_df,
        geometry='geometry',
        crs="EPSG:4326"
    )

    # Keep a separate county_gdf (with multi-polygons)
    county_gdf['centroid'] = county_gdf.geometry.centroid

    return hospital_gdf, county_gdf

hospital_gdf, county_gdf = load_data()

##############################################
# Build Hospital Lists by County
##############################################
@st.cache_data
def build_hospital_lists_by_county():
    """
    For each county, simply group hospital indices by county name.
    """
    result = {}
    for _, cty_row in county_gdf.iterrows():
        cty_name = cty_row['Name']
        cty_hosp = hospital_gdf[hospital_gdf['County'] == cty_name]
        result[cty_name] = list(cty_hosp.index)
    return result

hospitals_by_county = build_hospital_lists_by_county()

##############################################
# Simulation Helpers
##############################################
def update_hospital_occupancy(hospitals):
    """Compute used beds, then re-assign occupancy% for each row."""
    for idx, row in hospitals.iterrows():
        used = row['TOTAL BEDS, POS'] - row['Available Beds']
        if row['TOTAL BEDS, POS'] > 0:
            hospitals.at[idx, 'Occupancy'] = (used / row['TOTAL BEDS, POS']) * 100
        else:
            hospitals.at[idx, 'Occupancy'] = 100.0

def create_agents(ratio=100):
    """
    Create agents:
      1 agent per 'ratio' people from each county's population
      (taken from 'Total Population' in the shapefile).
    """
    agent_list = []
    for _, row in county_gdf.iterrows():
        pop_val = row['Total Population']
        cty_name = row['Name']
        for _ in range(pop_val // ratio):
            agent_list.append({
                'County': cty_name,
                'Infected': False,
                'Days Infected': 0,
                'Hospitalized': False,
                'Alive': True,
                'Traveled': False  # track if agent traveled => raised mortality
            })
    return agent_list

def run_one_day(
    agents,
    hospitals,
    infection_rate,
    base_mortality,
    travel_mortality_mult,
    min_ill,
    max_ill,
    traveled_count
):
    """
    Simulate a single day:
      - Infect new agents
      - Decrement infection timers
      - Attempt hospital assignment
      - Recompute hospital occupancy

    'travel_mortality_mult' is up to 5.0 for mortality if agent had to travel.
    """
    # 1) Infect or recover
    for agent in agents:
        if agent['Alive']:
            if not agent['Infected'] and random.random() < infection_rate:
                agent['Infected'] = True
                agent['Days Infected'] = random.randint(min_ill, max_ill)

            if agent['Infected']:
                agent['Days Infected'] -= 1
                if agent['Days Infected'] <= 0:
                    agent['Infected'] = False
                    eff_mortality = base_mortality
                    if agent['Traveled']:
                        eff_mortality *= travel_mortality_mult
                    if random.random() < eff_mortality:
                        agent['Alive'] = False

    # 2) Attempt hospital assignment
    for agent in agents:
        if agent['Alive'] and agent['Infected'] and not agent['Hospitalized']:
            cty = agent['County']
            if cty not in hospitals_by_county:
                continue
            sorted_hs = hospitals_by_county[cty]
            first_attempt = True
            for hidx in sorted_hs:
                if hospitals.at[hidx, 'Available Beds'] > 0:
                    hospitals.at[hidx, 'Available Beds'] -= 1
                    agent['Hospitalized'] = True
                    break
                else:
                    if first_attempt:
                        traveled_count += 1
                        agent['Traveled'] = True
                    first_attempt = False

    # 3) Recompute hospital occupancy
    update_hospital_occupancy(hospitals)

    return agents, hospitals, traveled_count

##############################################
# County-Level Stats
##############################################
def build_county_stats(hospitals, agents):
    """
    Build a DataFrame showing, for each county:
      - Population
      - Infected (agent count)
      - Total Beds
      - Used Beds
      - Occupancy %
    """
    agent_df = pd.DataFrame(agents)
    county_infection = agent_df[agent_df['Alive']].groupby('County')['Infected'].sum()

    hosp_df = hospitals.groupby('County').agg(
        TotalBeds=('TOTAL BEDS, POS', 'sum'),
        AvailableBeds=('Available Beds', 'sum')
    )
    hosp_df['UsedBeds'] = hosp_df['TotalBeds'] - hosp_df['AvailableBeds']
    hosp_df['Occupancy'] = hosp_df.apply(
        lambda row: (row['UsedBeds'] / row['TotalBeds']) * 100 if row['TotalBeds'] > 0 else 0,
        axis=1
    )

    pop_df = county_gdf[['Name','Total Population']].rename(columns={'Name':'County'})
    stats_df = pop_df.merge(hosp_df, on='County', how='left').fillna(0)

    stats_df = stats_df.merge(
        county_infection.rename('Infected'),
        on='County',
        how='left'
    ).fillna(0)

    stats_df = stats_df[[
        'County',
        'Total Population',
        'Infected',
        'TotalBeds',
        'UsedBeds',
        'Occupancy'
    ]].copy()

    stats_df['Occupancy'] = stats_df['Occupancy'].round(2)
    return stats_df

##############################################
# Color Counties by Occupancy (Matplotlib)
##############################################
def color_counties_map(hospitals, counties):
    """
    Re-project counties to EPSG:3857, compute county-level occupancy,
    color them green/yellow/orange/red, or black if no hospital.

    *Now includes a color legend.*
    """
    proj_counties = counties.to_crs(epsg=3857).copy()

    # usage_data => {county_name: (used_beds, total_beds)}
    usage_data = {}
    for _, row in hospitals.iterrows():
        cty = row['County']
        used = row['TOTAL BEDS, POS'] - row['Available Beds']
        tot  = row['TOTAL BEDS, POS']
        if cty not in usage_data:
            usage_data[cty] = [0,0]
        usage_data[cty][0] += used
        usage_data[cty][1] += tot

    proj_counties["occupancy"] = -1.0  # default to -1 => black if no hospital
    for i, row in proj_counties.iterrows():
        cty_name = row["Name"]
        if cty_name in usage_data:
            used, tot = usage_data[cty_name]
            if tot > 0:
                proj_counties.at[i, "occupancy"] = (used / tot) * 100
            else:
                proj_counties.at[i, "occupancy"] = -1

    color_col = []
    for i, row in proj_counties.iterrows():
        occ = row["occupancy"]
        if occ < 0:
            color_col.append("black")
        elif occ < 30:
            color_col.append("green")
        elif occ < 50:
            color_col.append("yellow")
        elif occ < 80:
            color_col.append("orange")
        else:
            color_col.append("red")
    proj_counties["color_col"] = color_col

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot subsets by color
    for colr in ["black", "green", "yellow", "orange", "red"]:
        subset = proj_counties[proj_counties["color_col"] == colr]
        if not subset.empty:
            subset.plot(ax=ax, color=colr, edgecolor="black")

    ax.set_title("New York State Counties: Hospital Occupancy")
    ax.axis("off")

    # --- Add a color legend ---
    legend_items = [
        mpatches.Patch(color='black',  label='No Hospital'),
        mpatches.Patch(color='green',  label='< 30%'),
        mpatches.Patch(color='yellow', label='30-50%'),
        mpatches.Patch(color='orange', label='50-80%'),
        mpatches.Patch(color='red',    label='>80%')
    ]
    ax.legend(
        handles=legend_items,
        loc='lower left',
        title='Occupancy Levels'
    )
    # --------------------------

    return fig

##############################################
# PDF Table + Image
##############################################
def create_pdf_report(lines, logo=None):
    """
    lines: list of (Metric, Value)
    If 'logo' is not None and the file exists, embed that image in the PDF.
    """
    pdf = FPDF()
    pdf.add_page()

    if logo and os.path.exists(logo):
        pdf.image(logo, x=65, y=10, w=80)
        pdf.ln(35)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Final Simulation Report", ln=1, align="C")
    pdf.ln(5)

    col_widths = [70, 70]
    for metric, val in lines:
        pdf.cell(col_widths[0], 8, metric, border=1)
        pdf.cell(col_widths[1], 8, val, border=1, ln=1)

    pdf_str = pdf.output(dest='S').encode("latin1")
    pdf_buf = io.BytesIO(pdf_str)
    pdf_buf.seek(0)
    return pdf_buf

##############################################
# Streamlit App
##############################################
st.title("Agent Based Modeling to Stress Testing New York State Healthcare Robustness (2020 Data)")

# Sidebar controls
infection_rate = st.sidebar.slider("Infection Probability", 0.0, 1.0, 0.25, 0.01)
base_mortality = st.sidebar.slider("Base Mortality Probability", 0.0, 1.0, 0.25, 0.01)
travel_mortality_mult = st.sidebar.slider("Travel Mortality Multiplier", 1.0, 5.0, 1.0, 0.1)
days = st.sidebar.number_input("Simulation Days", min_value=1, max_value=365, value=30)
illness_time = st.sidebar.slider("Illness Duration (Min,Max)", 1, 30, (6, 10))
start_occ = st.sidebar.slider("Starting Occupancy (%)", 0, 90, 0)
ratio = st.sidebar.number_input("Agents Ratio (1 agent per X people)", min_value=50, max_value=10000, value=100)

run_button = st.sidebar.button("Run Simulation")

if run_button:
    sim_hospitals = hospital_gdf.copy()

    # Optional: Apply starting occupancy
    if start_occ > 0:
        for idx, row in sim_hospitals.iterrows():
            used = int(row['TOTAL BEDS, POS'] * start_occ / 100)
            sim_hospitals.at[idx, 'Available Beds'] = row['TOTAL BEDS, POS'] - used
        update_hospital_occupancy(sim_hospitals)

    # Create agents
    sim_agents = create_agents(ratio=ratio)

    occupant_data = []
    traveled_count = 0
    full_capacity_set = set()

    chart_spot  = st.empty()
    map_spot    = st.empty()
    day_text    = st.empty()
    summary_spot= st.empty()
    county_table_spot = st.empty()
    progress    = st.progress(0)

    for day in range(1, days+1):
        sim_agents, sim_hospitals, traveled_count = run_one_day(
            sim_agents,
            sim_hospitals,
            infection_rate,
            base_mortality,
            travel_mortality_mult,
            illness_time[0],
            illness_time[1],
            traveled_count
        )

        avg_occ = sim_hospitals["Occupancy"].mean()
        occupant_data.append((day, avg_occ))

        for idx, row in sim_hospitals.iterrows():
            if row["Available Beds"] <= 0:
                full_capacity_set.add(idx)

        day_text.write(f"**Day {day}/{days}** - Avg Occupancy: {avg_occ:.2f}%")

        df_line = pd.DataFrame(occupant_data, columns=["Day", "Average Hospital Occupancy"])
        chart_spot.line_chart(df_line, x="Day", y="Average Hospital Occupancy")

        # Render map
        fig = color_counties_map(sim_hospitals, county_gdf)
        map_spot.pyplot(fig)

        tot_deaths = sum(not a["Alive"] for a in sim_agents)
        fc_count = len(full_capacity_set)
        total_fac = len(sim_hospitals)
        fc_pct = (fc_count / total_fac) * 100 if total_fac > 0 else 0
        all_full = (fc_count == total_fac)

        summary_lines = [
            ("Avg Occupancy", f"{avg_occ:.2f}%"),
            ("Total Deaths", f"{tot_deaths}"),
            ("Travel Mortality Mult", f"{travel_mortality_mult:.1f}"),
            ("Agents Who Traveled", f"{traveled_count}"),
            ("Facilities Full", f"{fc_count}/{total_fac} ({fc_pct:.1f}%)"),
            ("All Facilities Full?", f"{all_full}")
        ]
        df_summary = pd.DataFrame(summary_lines, columns=["Metric","Value"])
        summary_spot.table(df_summary)

        # County-level stats
        county_stats_df = build_county_stats(sim_hospitals, sim_agents)
        county_table_spot.write("#### County-Level Stats")
        county_table_spot.dataframe(county_stats_df)

        progress.progress(int(day/days*100))
        time.sleep(0.1)

    final_avg_occ = occupant_data[-1][1] if occupant_data else 0
    tot_deaths = sum(not a["Alive"] for a in sim_agents)
    fc_count = len(full_capacity_set)
    total_fac = len(sim_hospitals)
    fc_pct = (fc_count / total_fac) * 100 if total_fac > 0 else 0
    all_full = (fc_count == total_fac)

    st.write("## Final Summary Table")
    final_report_lines = [
        ("Final Average Occupancy", f"{final_avg_occ:.2f}%"),
        ("Total Deaths", f"{tot_deaths}"),
        ("Travel Mortality Multiplier", f"{travel_mortality_mult:.1f}"),
        ("Simulation Length (Days)", f"{days}"),
        ("Mortality Probability (Base)", f"{base_mortality:.2f}"),
        ("Agents Who Traveled", f"{traveled_count}"),
        ("Facilities Full", f"{fc_count}/{total_fac} ({fc_pct:.1f}%)"),
        ("All Facilities Full?", f"{all_full}")
    ]
    df_final = pd.DataFrame(final_report_lines, columns=["Metric","Value"])
    st.table(df_final)

    st.write("## Export PDF Report")
    pdf_buf = create_pdf_report(final_report_lines, logo=logo_path)
    st.download_button(
        label="Download PDF",
        data=pdf_buf,
        file_name="simulation_report.pdf",
        mime="application/pdf"
    )
else:
    st.write("Configure parameters in the sidebar, then click **Run Simulation**.")

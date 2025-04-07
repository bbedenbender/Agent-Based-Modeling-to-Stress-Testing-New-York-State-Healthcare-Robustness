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
logo_path = os.path.join(BASE_DIR, "agent.png")  # Place a PNG in the same folder (optional)

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
    hospital_data = hospital_data[hospital_data['Emergency Services'] == 'Yes']
    hospital_data = hospital_data[hospital_data['Federal'] != True]

    hospital_data['County'] = hospital_data['County'].str.title().str.strip()

    # Convert bed columns to numeric
    hospital_data['TOTAL BEDS, POS'] = pd.to_numeric(
        hospital_data['TOTAL BEDS, POS'], errors='coerce'
    ).fillna(0)
    hospital_data['Available Beds'] = hospital_data['TOTAL BEDS, POS']
    hospital_data['Occupancy'] = 0.0

    # 2) Load county shapefile
    county_gdf = gpd.read_file(county_shapefile_path).to_crs(epsg=4326)
    
    # Build neighbors for counties
    @st.cache_data
    def build_neighbors():
        neighbors = {}
        # Ensure county names are properly formatted
        county_gdf['Name'] = county_gdf['NAME'].str.title().str.strip()
        for idx1, row1 in county_gdf.iterrows():
            name1 = row1['Name']
            neighbors[name1] = []
            for idx2, row2 in county_gdf.iterrows():
                name2 = row2['Name']
                if name1 != name2 and row1.geometry.touches(row2.geometry):
                    neighbors[name1].append(name2)
        return neighbors
    county_neighbors = build_neighbors()

    # Ensure county names match for hospital data and county shapefile
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

    # Compute county centroids
    county_gdf['centroid'] = county_gdf.geometry.centroid

    return hospital_gdf, county_gdf, county_neighbors

hospital_gdf, county_gdf, county_neighbors = load_data()

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



def create_agents(ratio=100, seed_county="Albany", seed_infections=10):
    agent_list = []
    for _, row in county_gdf.iterrows():
        pop_val = row['Total Population']
        cty_name = row['Name']
        agents_in_county = []
        for _ in range(pop_val // ratio):
            agents_in_county.append({
                'County': cty_name,
                'Infected': False,
                'Days Infected': 0,
                'Hospitalized': False,
                'Alive': True,
                'Traveled': False,
                'Recovered': False  # <-- NEW ATTRIBUTE
            })
        if cty_name == seed_county:
            infected_indices = random.sample(range(len(agents_in_county)), min(seed_infections, len(agents_in_county)))
            for idx in infected_indices:
                agents_in_county[idx]['Infected'] = True
                agents_in_county[idx]['Days Infected'] = random.randint(6, 10)
        agent_list.extend(agents_in_county)
    return agent_list


def run_one_day(agents, hospitals, infection_rate, base_mortality, travel_mortality_mult, 
                min_ill, max_ill, traveled_count, trip_count, reinfection_rate, hospitalization_likelihood):
    """
    Simulate one day of the model, considering reinfection_rate and illness severity.
    Returns agents, hospitals, traveled_count, and trip_count.
    """
    # 1) Count infections by county
    county_infection_counts = {cty: 0 for cty in county_gdf['Name']}
    for agent in agents:
        if agent['Alive'] and agent['Infected']:
            county_infection_counts[agent['County']] += 1

    # 2) Spread infection (new infections or reinfections)
    for agent in agents:
        if agent['Alive'] and not agent['Infected']:
            cty = agent['County']
            neighbors = county_neighbors.get(cty, [])
            exposure = county_infection_counts.get(cty, 0)
            for neighbor in neighbors:
                exposure += county_infection_counts.get(neighbor, 0)
            if exposure > 0:
                probability = infection_rate * min(1, exposure / 10)
                if agent['Recovered']:
                    probability *= reinfection_rate
                if random.random() < probability:
                    agent['Infected'] = True
                    agent['Days Infected'] = random.randint(min_ill, max_ill)
                    agent['Recovered'] = False

    # 3) Update infection duration, check for recovery/death
    for agent in agents:
        if agent['Alive'] and agent['Infected']:
            agent['Days Infected'] -= 1
            if agent['Days Infected'] <= 0:
                agent['Infected'] = False
                eff_mortality = base_mortality * (travel_mortality_mult if agent['Traveled'] else 1.0)
                if random.random() < eff_mortality:
                    agent['Alive'] = False
                else:
                    agent['Recovered'] = True

                # Release bed if hospitalized: use admitted county if available.
                if agent['Hospitalized']:
                    release_county = agent.get('AdmittedCounty', agent['County'])
                    for hidx in hospitals_by_county.get(release_county, []):
                        if hospitals.at[hidx, 'Available Beds'] < hospitals.at[hidx, 'TOTAL BEDS, POS']:
                            hospitals.at[hidx, 'Available Beds'] += 1
                            agent['Hospitalized'] = False
                            # Clear admitted county field after releasing bed.
                            agent.pop('AdmittedCounty', None)
                            break

    # 4) Attempt hospital assignment based on severity
    for agent in agents:
        if agent['Alive'] and agent['Infected'] and not agent['Hospitalized']:
            if random.random() < hospitalization_likelihood:
                # Create list of candidate counties: home county plus neighbors.
                candidate_counties = [agent['County']] + county_neighbors.get(agent['County'], [])
                bed_found = False
                max_attempts = 5  # Maximum search attempts per day.
                attempt = 0
                admitted_county = None
                while not bed_found and attempt < max_attempts:
                    for cty in candidate_counties:
                        if cty in hospitals_by_county:
                            for hidx in hospitals_by_county[cty]:
                                if hospitals.at[hidx, 'Available Beds'] > 0:
                                    hospitals.at[hidx, 'Available Beds'] -= 1
                                    agent['Hospitalized'] = True
                                    bed_found = True
                                    admitted_county = cty  # Record the county where the bed was found.
                                    break
                            if bed_found:
                                break
                    attempt += 1

                # Count unsuccessful trips for this day.
                # If a bed is found, subtract one attempt (the successful one) from the total.
                if bed_found:
                    unsuccessful_attempts = attempt - 1
                else:
                    unsuccessful_attempts = attempt

                # Update the agent's individual unsuccessful trips counter.
                agent['UnsuccessfulTrips'] = agent.get('UnsuccessfulTrips', 0) + unsuccessful_attempts

                # Also update the global trip count.
                trip_count += unsuccessful_attempts

                # If a bed is found in a county different from the agent's home county, count as travel.
                if bed_found:
                    if admitted_county != agent['County']:
                        if not agent['Traveled']:
                            traveled_count += 1
                            agent['Traveled'] = True
                    # Record the admitting county.
                    agent['AdmittedCounty'] = admitted_county

    # 5) Recompute hospital occupancy
    update_hospital_occupancy(hospitals)

    return agents, hospitals, traveled_count, trip_count

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
    """
    proj_counties = counties.to_crs(epsg=3857).copy()

    # usage_data => {county_name: (used_beds, total_beds)}
    usage_data = {}
    for _, row in hospitals.iterrows():
        cty = row['County']
        used = row['TOTAL BEDS, POS'] - row['Available Beds']
        tot  = row['TOTAL BEDS, POS']
        if cty not in usage_data:
            usage_data[cty] = [0, 0]
        usage_data[cty][0] += used
        usage_data[cty][1] += tot

        proj_counties["occupancy"] = -1.0
        
        # Compute occupancy
        for i, row in proj_counties.iterrows():
            cty_name = row["Name"]
            if cty_name in usage_data:
                used, tot = usage_data[cty_name]
                if tot > 0:
                    proj_counties.at[i, "occupancy"] = (used / tot) * 100
                else:
                    proj_counties.at[i, "occupancy"] = -1
        
        # Define color based on occupancy
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
            elif occ < 100:
                color_col.append("red")
            else:  # exactly 100%
                color_col.append("darkred")
        
        proj_counties["color_col"] = color_col


    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot subsets by color
    for colr in ["black", "green", "yellow", "orange", "red", "darkred"]:
        subset = proj_counties[proj_counties["color_col"] == colr]
        if not subset.empty:
            subset.plot(ax=ax, color=colr, edgecolor="black")


    ax.set_title("New York State Counties: Hospital Occupancy")
    ax.axis("off")

    legend_items = [
        mpatches.Patch(color='black',  label='No Hospital'),
        mpatches.Patch(color='green',  label='< 30%'),
        mpatches.Patch(color='yellow', label='30-50%'),
        mpatches.Patch(color='orange', label='50-80%'),
        mpatches.Patch(color='red',    label='80-99%'),
        mpatches.Patch(color='darkred',label='100% (Full Capacity)')
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
        pdf.cell(col_widths[1], 8, str(val), border=1, ln=1)

    pdf_str = pdf.output(dest='S').encode("latin1")
    pdf_buf = io.BytesIO(pdf_str)
    pdf_buf.seek(0)
    return pdf_buf

##############################################
# Streamlit App
##############################################


st.title("Agent-Based Modeling to Stress Test New York State Healthcare Robustness (2020 Data)")

# Sidebar controls
# Display logo at top of sidebar
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.sidebar.image(logo, width=50)  # Adjust width as needed


infection_rate = st.sidebar.slider("Agent Infection Probability", 0.0, 1.0, 0.25, 0.01)
hospitalization_likelihood = st.sidebar.slider(
    "Illness Severity Requiring Hospitalization (%)", 
    min_value=0, max_value=100, value=20, step=1
) / 100  # convert to decimal

reinfection_rate = st.sidebar.select_slider(
    "Agent Reinfection Probability (%)",
    options=[0, 10, 25, 50, 75, 100],
    value=25
) / 100  # convert percentage to a decimal

base_mortality = st.sidebar.slider("Base Mortality Probability", 0.0, 1.0, 0.25, 0.01)
travel_mortality_mult = st.sidebar.slider("Travel Mortality Multiplier", 1.0, 5.0, 1.0, 0.1)
days = st.sidebar.number_input("Simulation Days", min_value=1, max_value=1825, value=30)
illness_time = st.sidebar.slider("Illness Duration Day(s) (Min,Max)", 1, 30, (6, 10))
start_occ = st.sidebar.slider("Starting Hospital Occupancy (%)", 0, 90, 0)
ratio = st.sidebar.number_input("Agents Ratio (1 agent per X people)", min_value=50, max_value=10000, value=100)

# Build a dropdown OR choose random seed county
random_seed = st.sidebar.checkbox("Use Random Seed County", value=False)
county_names = county_gdf['Name'].sort_values().tolist()
if random_seed:
    seed_county = random.choice(county_names)
    st.sidebar.write(f"Randomly selected county: **{seed_county}**")
else:
    seed_county = st.sidebar.selectbox("Select Seed County", county_names)

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
    sim_agents = create_agents(ratio=ratio, seed_county=seed_county, seed_infections=10)

    occupant_data = []
    traveled_count = 0
    full_capacity_set = set()

    chart_spot  = st.empty()
    map_spot    = st.empty()
    day_text    = st.empty()
    summary_spot= st.empty()
    county_table_spot = st.empty()
    progress    = st.progress(0)
    traveled_count = 0
    trip_count = 0

    for day in range(1, days + 1):
        sim_agents, sim_hospitals, traveled_count, trip_count = run_one_day(
            sim_agents,
            sim_hospitals,
            infection_rate,
            base_mortality,
            travel_mortality_mult,
            illness_time[0],
            illness_time[1],
            traveled_count,
            trip_count,
            reinfection_rate,
            hospitalization_likelihood  
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
            ("Agents Who Traveled for Care", f"{traveled_count}"),
            ("Percent of Facilities Reaching Full Capacity", f"{fc_count}/{total_fac} ({fc_pct:.1f}%)"),
            ("Did All Facilities Reach Full Capacity?", f"{all_full}")
        ]
        df_summary = pd.DataFrame(summary_lines, columns=["Metric", "Value"])
        summary_spot.table(df_summary)

        # County-level stats
        county_stats_df = build_county_stats(sim_hospitals, sim_agents)
        county_table_spot.write("#### County-Level Stats")
        county_table_spot.dataframe(county_stats_df)

        progress.progress(int(day / days * 100))
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
        ("Reinfection Probability", f"{reinfection_rate*100:.0f}%"),  # <-- Added
        ("Agents Who Traveled for Care", f"{traveled_count}"),
        ("Percent of Facilities Reaching Full Capacity", f"{fc_count}/{total_fac} ({fc_pct:.1f}%)"),
        ("Did All Facilities Reach Full Capacity?", f"{all_full}")
    ]

    df_final = pd.DataFrame(final_report_lines, columns=["Metric", "Value"])
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

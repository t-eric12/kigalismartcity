import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor  # type: ignore # ✅ Add this line
from sklearn.model_selection import train_test_split  # type: ignore # ✅ Add this line
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
import requests
import networkx as nx  # ✅ Add this line

# Set page config
st.set_page_config(
    page_title="Kigali Smart Traffic Control System",
    page_icon="🚦",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: white;
        background: linear-gradient(to right, #1a3b88, #00853e);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }
             /* Center align Streamlit tabs */
    div[data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
    }
    .card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .card-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .card-subtitle {
        font-size: 0.9rem;
        color: #6b7280;
    }
    .status-high {
        color: #ef4444;
    }
    .status-medium {
        color: #f59e0b;
    }
    .status-low {
        color: #10b981;
    }
    .event-title {
        font-weight: 600;
        font-size: 1rem;
    }
    .event-location, .event-time {
        font-size: 0.8rem;
        color: #6b7280;
    }
    .event-impact {
        font-size: 0.8rem;
        color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Kigali Smart Traffic Control System</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Intersection Details", "Simulation", "Route Planner", "Special Events Calendar"])

# Generate sample data
def generate_traffic_data():
    hours = list(range(0, 24))
    traffic_values = [1200, 800, 500, 300, 400, 1000, 2500, 3500, 3200, 2800, 
                     2700, 2900, 2800, 2700, 2900, 3200, 3800, 3500, 3000, 
                     2500, 2200, 1800, 1500, 1300]
    return pd.DataFrame({'hour': hours, 'volume': traffic_values})

def generate_flow_data():
    directions = ['North', 'South', 'East', 'West']
    times = pd.date_range(datetime.now() - timedelta(hours=8), periods=8, freq='H')
    data = []
    
    for time in times:
        for direction in directions:
            base_flow = {'North': 150, 'South': 180, 'East': 120, 'West': 100}
            hour_factor = {
                6: 1.2, 7: 1.8, 8: 2.0, 9: 1.7, 
                10: 1.5, 11: 1.4, 12: 1.6, 13: 1.5,
                14: 1.4, 15: 1.5, 16: 1.8, 17: 2.2,
                18: 1.9, 19: 1.6, 20: 1.3, 21: 1.1,
                22: 0.9, 23: 0.7, 0: 0.5, 1: 0.3,
                2: 0.2, 3: 0.2, 4: 0.3, 5: 0.8
            }
            flow = base_flow[direction] * hour_factor[time.hour] * (1 + 0.2 * np.random.randn())
            data.append({
                'time': time,
                'direction': direction,
                'flow': max(0, flow)
            })
    
    return pd.DataFrame(data)

def generate_congestion_forecast():
    hours = list(range(1, 13))
    now = datetime.now()
    times = [(now + timedelta(hours=h)).strftime("%H:%M") for h in hours]
    
    areas = ['Downtown', 'Airport Area', 'Convention Centre', 'Residential Areas']
    data = []
    
    for area in areas:
        base_congestion = {
            'Downtown': 75, 
            'Airport Area': 60, 
            'Convention Centre': 85, 
            'Residential Areas': 40
        }
        
        for i, time in enumerate(times):
            hour = (now.hour + i + 1) % 24
            time_factor = {
                6: 0.8, 7: 1.2, 8: 1.5, 9: 1.3, 
                10: 1.0, 11: 0.9, 12: 1.1, 13: 1.2,
                14: 1.0, 15: 1.1, 16: 1.3, 17: 1.5,
                18: 1.4, 19: 1.2, 20: 0.9, 21: 0.7,
                22: 0.6, 23: 0.5, 0: 0.3, 1: 0.2,
                2: 0.1, 3: 0.1, 4: 0.2, 5: 0.5
            }
            
            congestion = base_congestion[area] * time_factor.get(hour, 1.0) * (1 + 0.15 * np.random.randn())
            congestion = max(0, min(100, congestion))
            
            data.append({
                'time': time,
                'area': area,
                'congestion': congestion
            })
    
    return pd.DataFrame(data)

traffic_data = generate_traffic_data()
flow_data = generate_flow_data()
congestion_data = generate_congestion_forecast()

# Overview Tab
with tab1:
    # Status Cards Row
        status_cols = st.columns(3)
    
with status_cols[0]:
        st.markdown("""
        <div class="card">
            <h3 class="card-title">Current Traffic Status</h3>
            <p class="card-value">Moderate</p>
            <p class="card-subtitle">25% below average for Friday</p>
        </div>
        """, unsafe_allow_html=True)
    
with status_cols[1]:
        st.markdown("""
        <div class="card">
            <h3 class="card-title">Weather Impact</h3>
            <p class="card-value">Low</p>
            <p class="card-subtitle">Clear skies, 24°C</p>
        </div>
        """, unsafe_allow_html=True)
    
with status_cols[2]:
        st.markdown("""
        <div class="card">
            <h3 class="card-title">System Alerts</h3>
            <p class="card-value">2</p>
            <p class="card-subtitle">Camera maintenance required</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts Row
        chart_cols = st.columns(2)
    
with chart_cols[0]:
        st.markdown('<div class="card-title">Traffic Volume (Last 24 Hours)</div>', unsafe_allow_html=True)
        fig = px.line(traffic_data, x='hour', y='volume', 
                     labels={'hour': 'Hour of Day', 'volume': 'Number of Vehicles'},
                     height=300)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
with chart_cols[1]:
        st.markdown('<div class="card-title">Congested Intersections</div>', unsafe_allow_html=True)
        
        intersections = [
            {"name": "KN 5 Rd / KG 7 Ave", "status": "High Volume", "level": "high"},
            {"name": "KN 3 Rd / KG 15 Ave", "status": "Medium Volume", "level": "medium"},
            {"name": "Convention Centre Roundabout", "status": "Medium Volume", "level": "medium"}
        ]
        
        for intersection in intersections:
            level_class = f"status-{intersection['level']}"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.75rem; border: 1px solid #e5e7eb; border-radius: 0.25rem; margin-bottom: 0.5rem;">
                <div>
                    <span class="{level_class}">●</span> {intersection["name"]}
                </div>
                <div style="color: #6b7280; font-size: 0.9rem;">
                    {intersection["status"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    


# Intersection Details Tab
with tab2:
         intersection_cols = st.columns([1.5, 1])
    
with intersection_cols[0]:
        st.markdown('<div class="card-title">Intersection Map</div>', unsafe_allow_html=True)
        
        # Create a map centered on Kigali
        m = folium.Map(location=[-1.9415, 30.0415], zoom_start=14)
        
        # Add markers for key intersections
        intersections = [
            {"name": "KN 5 Rd / KG 7 Ave", "location": [-1.9500, 30.0600], "status": "high"},
            {"name": "KN 3 Rd / KG 15 Ave", "location": [-1.9450, 30.0500], "status": "medium"},
            {"name": "Convention Centre Roundabout", "location": [-1.9550, 30.0700], "status": "medium"},
            {"name": "Airport Roundabout", "location": [-1.9700, 30.1300], "status": "low"},
            {"name": "Kimihurura Roundabout", "location": [-1.9600, 30.0900], "status": "low"}
        ]
        
        color_map = {"high": "red", "medium": "orange", "low": "green"}
        
        for intersection in intersections:
            folium.CircleMarker(
                location=intersection["location"],
                radius=8,
                color=color_map[intersection["status"]],
                fill=True,
                fill_color=color_map[intersection["status"]],
                tooltip=intersection["name"]
            ).add_to(m)
        
        # Display the map
        folium_static(m, width=600, height=350)
    
with intersection_cols[1]:
        st.markdown('<div class="card-title">Intersection Details</div>', unsafe_allow_html=True)
        
        selected_intersection = st.selectbox(
            "Select Intersection",
            ["KN 5 Rd / KG 7 Ave", "KN 3 Rd / KG 15 Ave", "Convention Centre Roundabout", 
             "Airport Roundabout", "Kimihurura Roundabout", 'KN 1 St / KN 2 Ave', 'KN 3 Rd / KN 4 Ave', 'KN 5 Rd / KG 7 Ave', 'KN 6 Ave / KG 8 Ave',
            'KN 7 Rd / KG 9 Ave', 'KN 8 Ave / KG 10 Ave', 'KN 9 Rd / KG 11 Ave', 'KN 10 Ave / KG 12 Ave',
            'KN 11 Rd / KG 13 Ave', 'KN 12 Ave / KG 14 Ave', 'KN 13 Rd / KG 15 Ave', 'KN 14 Ave / KG 16 Ave',
            'KN 15 Rd / KG 17 Ave', 'KN 16 Ave / KG 18 Ave', 'KN 17 Rd / KG 19 Ave', 'KN 18 Ave / KG 20 Ave',
            'KN 19 Rd / KG 21 Ave', 'KN 20 Ave / KG 22 Ave', 'KN 21 Rd / KG 23 Ave', 'KN 22 Ave / KG 24 Ave',
            'KN 23 Rd / KG 25 Ave', 'KN 24 Ave / KG 26 Ave', 'KN 25 Rd / KG 27 Ave', 'KN 26 Ave / KG 28 Ave',
            'KN 27 Rd / KG 29 Ave', 'KN 28 Ave / KG 30 Ave', 'KN 29 Rd / KG 31 Ave', 'KN 30 Ave / KG 32 Ave',
            'KN 31 Rd / KG 33 Ave', 'KN 32 Ave / KG 34 Ave', 'KN 33 Rd / KG 35 Ave', 'KN 34 Ave / KG 36 Ave',
            'KN 35 Rd / KG 37 Ave', 'KN 36 Ave / KG 38 Ave', 'KN 37 Rd / KG 39 Ave', 'KN 38 Ave / KG 40 Ave',
            'KN 39 Rd / KG 41 Ave', 'KN 40 Ave / KG 42 Ave', 'KN 41 Rd / KG 43 Ave', 'KN 42 Ave / KG 44 Ave',
            'KN 43 Rd / KG 45 Ave', 'KN 44 Ave / KG 46 Ave', 'KN 45 Rd / KG 47 Ave', 'KN 46 Ave / KG 48 Ave',
            'KN 47 Rd / KG 49 Ave', 'KN 48 Ave / KG 50 Ave', 'KN 49 Rd / KG 51 Ave', 'KN 50 Ave / KG 52 Ave']
             
        )
        
        status_map = {
            "KN 5 Rd / KG 7 Ave": {"status": "High Volume", "level": "high", "cycle": 120},
            "KN 3 Rd / KG 15 Ave": {"status": "Medium Volume", "level": "medium", "cycle": 90},
            "Convention Centre Roundabout": {"status": "Medium Volume", "level": "medium", "cycle": 100},
            "Airport Roundabout": {"status": "Low Volume", "level": "low", "cycle": 60},
            "Kimihurura Roundabout": {"status": "Low Volume", "level": "low", "cycle": 70},
            'KN 1 St / KN 2 Ave': {"status": "Low Volume", "level": "low", "cycle": 60},
            'KN 3 Rd / KN 4 Ave': {"status": "Low Volume", "level": "low", "cycle": 70},
            'KN 5 Rd / KG 7 Ave': {"status": "High Volume", "level": "high", "cycle": 120},
            'KN 6 Ave / KG 8 Ave': {"status": "High Volume", "level": "high", "cycle": 110},
            'KN 7 Rd / KG 9 Ave': {"status": "Medium Volume", "level": "medium", "cycle": 90},
            'KN 8 Ave / KG 10 Ave': {"status": "Medium Volume", "level": "medium", "cycle": 100},
            'KN 9 Rd / KG 11 Ave': {"status": "Low Volume", "level": "low", "cycle": 60},
            'KN 10 Ave / KG 12 Ave': {"status": "Low Volume", "level": "low", "cycle": 70},
            'KN 11 Rd / KG 13 Ave': {"status": "High Volume", "level": "high", "cycle": 120},
            'KN 12 Ave / KG 14 Ave': {"status": "High Volume", "level": "high", "cycle": 110},
            'KN 13 Rd / KG 15 Ave': {"status": "Medium Volume", "level": "medium", "cycle": 90},
            'KN 14 Ave / KG 16 Ave': {"status": "Medium Volume", "level": "medium", "cycle": 100},
            'KN 15 Rd / KG 17 Ave': {"status": "Low Volume", "level": "low", "cycle": 60},
            'KN 16 Ave / KG 18 Ave': {"status": "Low Volume", "level": "low", "cycle": 70},
            'KN 17 Rd / KG 19 Ave': {"status": "High Volume", "level": "high", "cycle": 120},
            'KN 18 Ave / KG 20 Ave': {"status": "High Volume", "level": "high", "cycle": 110},
            'KN 19 Rd / KG 21 Ave': {"status": "Medium Volume", "level": "medium", "cycle": 90},
            'KN 20 Ave / KG 22 Ave': {"status": "Medium Volume", "level": "medium", "cycle": 100},
            'KN 21 Rd / KG 23 Ave': {"status": "Low Volume", "level": "low", "cycle": 60},
            'KN 22 Ave / KG 24 Ave': {"status": "Low Volume", "level": "low", "cycle": 70},
            'KN 23 Rd / KG 25 Ave': {"status": "High Volume", "level": "high", "cycle": 120},
            'KN 24 Ave / KG 26 Ave': {"status": "High Volume", "level": "high", "cycle": 110},
            'KN 25 Rd / KG 27 Ave': {"status": "Medium Volume", "level": "medium", "cycle": 90},
            'KN 26 Ave / KG 28 Ave': {"status": "Medium Volume", "level": "medium", "cycle": 100},
            'KN 27 Rd / KG 29 Ave': {"status": "Low Volume", "level": "low", "cycle": 60},
            'KN 28 Ave / KG 30 Ave': {"status": "Low Volume", "level": "low", "cycle": 70},
            'KN 29 Rd / KG 31 Ave': {"status": "High Volume", "level": "high", "cycle": 120},
            'KN 30 Ave / KG 32 Ave': {"status": "High Volume", "level": "high", "cycle": 110},
            'KN 31 Rd / KG 33 Ave': {"status": "Medium Volume", "level": "medium", "cycle": 90},
            'KN 32 Ave / KG 34 Ave': {"status": "Medium Volume", "level": "medium", "cycle": 100},
            'KN 33 Rd / KG 35 Ave': {"status": "Low Volume", "level": "low", "cycle": 60},
            'KN 34 Ave / KG 36 Ave': {"status": "Low Volume", "level": "low", "cycle": 70},
            'KN 35 Rd / KG 37 Ave': {"status": "High Volume", "level": "high", "cycle": 120},
            'KN 36 Ave / KG 38 Ave': {"status": "High Volume", "level": "high", "cycle": 110},
            'KN 37 Rd / KG 39 Ave': {"status": "Medium Volume", "level": "medium", "cycle": 90},
            'KN 38 Ave / KG 40 Ave': {"status": "Medium Volume", "level": "medium", "cycle": 100},
            'KN 39 Rd / KG 41 Ave': {"status": "Low Volume", "level": "low", "cycle": 60},
            'KN 40 Ave / KG 42 Ave': {"status": "Low Volume", "level": "low", "cycle": 70},
            'KN 41 Rd / KG 43 Ave': {"status": "High Volume", "level": "high", "cycle": 120},
            'KN 42 Ave / KG 44 Ave': {"status": "High Volume", "level": "high", "cycle": 110},
            'KN 43 Rd / KG 45 Ave': {"status": "Medium Volume", "level": "medium", "cycle": 90},
            'KN 44 Ave / KG 46 Ave': {"status": "Medium Volume", "level": "medium", "cycle": 100},
            'KN 45 Rd / KG 47 Ave': {"status": "Low Volume", "level": "low", "cycle": 60},
            'KN 46 Ave / KG 48 Ave': {"status": "Low Volume", "level": "low", "cycle": 70},
            'KN 47 Rd / KG 49 Ave': {"status": "High Volume", "level": "high", "cycle": 120},
            'KN 48 Ave / KG 50 Ave': {"status": "High Volume", "level": "high", "cycle": 110},
            'KN 49 Rd / KG 51 Ave': {"status": "Medium Volume", "level": "medium", "cycle": 90},
            'KN 50 Ave / KG 52 Ave': {"status": "Medium Volume", "level": "medium", "cycle": 100}
            
        }
        
        current_status = status_map[selected_intersection]
        level_class = f"status-{current_status['level']}"
        
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <span style="display: block; font-size: 0.875rem; font-weight: 500; color: #374151; margin-bottom: 0.25rem;">Current Status</span>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
                <span class="{level_class}">●</span>
                <span>{current_status["status"]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        traffic_mode = st.selectbox(
            "Traffic Light Mode",
            ["Adaptive", "Fixed", "Manual"]
        )
        st.markdown('<div class="card-title">Intersection Routes</div>', unsafe_allow_html=True)
        
       
        
# Simulated Traffic Data Generation
with tab3:
   def generate_traffic_data():
    intersections = ['KN 5 Rd / KG 7 Ave', 'KN 3 Rd / KG 15 Ave', 'Convention Centre', 'Airport Roundabout'
            'KN 1 St / KN 2 Ave', 'KN 3 Rd / KN 4 Ave', 'KN 6 Ave / KG 8 Ave',
            'KN 7 Rd / KG 9 Ave', 'KN 8 Ave / KG 10 Ave', 'KN 9 Rd / KG 11 Ave', 'KN 10 Ave / KG 12 Ave',
            'KN 11 Rd / KG 13 Ave', 'KN 12 Ave / KG 14 Ave', 'KN 13 Rd / KG 15 Ave', 'KN 14 Ave / KG 16 Ave',
            'KN 15 Rd / KG 17 Ave', 'KN 16 Ave / KG 18 Ave', 'KN 17 Rd / KG 19 Ave', 'KN 18 Ave / KG 20 Ave',
            'KN 19 Rd / KG 21 Ave', 'KN 20 Ave / KG 22 Ave', 'KN 21 Rd / KG 23 Ave', 'KN 22 Ave / KG 24 Ave',
            'KN 23 Rd / KG 25 Ave', 'KN 24 Ave / KG 26 Ave', 'KN 25 Rd / KG 27 Ave', 'KN 26 Ave / KG 28 Ave',
            'KN 27 Rd / KG 29 Ave', 'KN 28 Ave / KG 30 Ave', 'KN 29 Rd / KG 31 Ave', 'KN 30 Ave / KG 32 Ave',
            'KN 31 Rd / KG 33 Ave', 'KN 32 Ave / KG 34 Ave', 'KN 33 Rd / KG 35 Ave', 'KN 34 Ave / KG 36 Ave',
            'KN 35 Rd / KG 37 Ave', 'KN 36 Ave / KG 38 Ave', 'KN 37 Rd / KG 39 Ave', 'KN 38 Ave / KG 40 Ave',
            'KN 39 Rd / KG 41 Ave', 'KN 40 Ave / KG 42 Ave', 'KN 41 Rd / KG 43 Ave', 'KN 42 Ave / KG 44 Ave',
            'KN 43 Rd / KG 45 Ave', 'KN 44 Ave / KG 46 Ave', 'KN 45 Rd / KG 47 Ave', 'KN 46 Ave / KG 48 Ave',
            'KN 47 Rd / KG 49 Ave', 'KN 48 Ave / KG 50 Ave', 'KN 49 Rd / KG 51 Ave', 'KN 50 Ave / KG 52 Ave']
        
        
   
    data = []
    for _ in range(100):  # Generate 100 data points
        data.append({
            'intersection': np.random.choice(intersections),
            'traffic_volume': np.random.randint(50, 500),
            'time_of_day': np.random.randint(0, 24),
            'weather': np.random.choice(['Clear', 'Rainy', 'Cloudy']),
            'signal_time': np.random.randint(30, 120)
        })
    return pd.DataFrame(data)

# Train AI Model to Predict Optimal Signal Time
   def train_model(df):
    X = df[['traffic_volume', 'time_of_day']]
    y = df['signal_time']
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

# Streamlit UI
   st.title('🚦 AI-Powered Traffic Control Simulation')

   st.subheader('Simulation Controls')
   simulation_speed = st.slider('Simulation Speed (sec)', 1, 5, 2)

# Generate data & Train Model
   traffic_data = generate_traffic_data()
   model = train_model(traffic_data)

# Display Initial Traffic Data
   st.subheader('📊 Traffic Data Sample')
   st.dataframe(traffic_data.head(10))

# Live Traffic Simulation
   st.subheader('🔴🟡🟢 Real-Time Traffic Simulation')
   sim_data = []

   traffic_placeholder = st.empty()

   for _ in range(20):  # Simulate 20 iterations
        intersection = np.random.choice(traffic_data['intersection'].unique())
        traffic_volume = np.random.randint(50, 500)
        time_of_day = np.random.randint(0, 24)
    
    # AI Prediction
        predicted_signal_time = model.predict([[traffic_volume, time_of_day]])[0]
        sim_data.append({'intersection': intersection, 'traffic_volume': traffic_volume, 'signal_time': int(predicted_signal_time)})
    
    # Display simulation
        df_sim = pd.DataFrame(sim_data)
        fig = px.bar(df_sim, x='intersection', y='signal_time', color='traffic_volume',
                 labels={'signal_time': 'Green Light Duration (sec)', 'traffic_volume': 'Traffic Volume'})
        traffic_placeholder.plotly_chart(fig)
    
        time.sleep(simulation_speed)

        st.success('✅ Simulation Complete!')

# Route Planner
# Initialize geolocator
geolocator = Nominatim(user_agent="route_planner")

def get_coordinates(place):
    """Get latitude and longitude of a place using geopy."""
    try:
        location = geolocator.geocode(place)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        st.error(f"Geocoding error: {e}")
    return None
    

with tab4:
    st.title("🚗 Route Planner")


    # User Inputs
    origin = st.text_input("Enter Starting Point:")
    destination = st.text_input("Enter Destination:")

    transport_mode = st.selectbox(
        "Select Transport Mode",
        ["Driving", "Public Transport", "Walking", "Bicycling"],
        index=0
    )

    # Transport mode mapping
    mode_mapping = {
        "Driving": "driving",
        "Public Transport": "transit",
        "Walking": "walking",
        "Bicycling": "bicycling"
    }
    mode = mode_mapping[transport_mode]

    # Route Map
    if st.button("Find Route"):
        origin_coords = get_coordinates(origin)
        destination_coords = get_coordinates(destination)

        if origin_coords and destination_coords:
            # Display Map
            m = folium.Map(location=origin_coords, zoom_start=12)
            folium.Marker(origin_coords, popup="Origin", icon=folium.Icon(color="green")).add_to(m)
            folium.Marker(destination_coords, popup="Destination", icon=folium.Icon(color="red")).add_to(m)
            folium.PolyLine([origin_coords, destination_coords], color="blue", weight=5, opacity=0.7).add_to(m)
            
            folium_static(m)

            # Google Maps API for Route Details (Optional)
            GOOGLE_MAPS_API_KEY = "YOUR_API_KEY"  # Replace with your API Key
            url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&mode={mode}&key={GOOGLE_MAPS_API_KEY}"
            response = requests.get(url).json()

            if response["status"] == "OK":
                route = response["routes"][0]["legs"][0]
                distance = route["distance"]["text"]
                duration = route["duration"]["text"]

                st.subheader("📍 Route Details")
                st.write(f"**Estimated Time:** {duration}")
                st.write(f"**Distance:** {distance}")
            else:
                st.error("Could not retrieve route details. Check API key or locations.")
        else:
            st.error("Invalid location. Please enter a valid city or address.")

# Special Events Section
with tab5:
    st.subheader("🎉 Special Events Calendar")

    events = [
        {
            "title": "Kigali International Conference",
            "location": "Kigali Convention Centre",
            "time": "Today, 10:00 - 18:00",
            "impact": "High traffic expected around KN 5 Rd",
            "options": ["Shuttle Bus", "Alternative Routes"]
        },
        {
            "title": "Kigali Jazz Festival",
            "location": "Amahoro Stadium",
            "time": "Tomorrow, 16:00 - 22:00",
            "impact": "Moderate traffic expected around KN 3 Rd",
            "options": ["Public Transport", "Park & Ride"]
        }
    ]

    for event in events:
        st.markdown(f"""
        <div style="padding: 10px; margin: 10px 0; border-radius: 8px; background-color: #f9fafb; border-left: 5px solid #2563eb;">
            <h4 style="margin-bottom: 5px;">{event["title"]}</h4>
            <p><strong>📍 Location:</strong> {event["location"]}</p>
            <p><strong>🕒 Time:</strong> {event["time"]}</p>
            <p><strong>🚦 Traffic Impact:</strong> {event["impact"]}</p>
            <p><strong>🛑 Alternative Options:</strong> {", ".join(event["options"])}</p>
        </div>
        """, unsafe_allow_html=True)
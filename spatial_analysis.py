import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.metrics.pairwise import cosine_similarity
import folium
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np


# ======================
# 1. Load and prepare data
# ======================
@st.cache_data
def load_data():
    data = pd.read_excel("ozurgeti_dataset_coordinates_filtered.xlsx")
    nace_rev2 = pd.read_excel("NACE Rev2.xlsx")

    # Split NACE Rev.2 code
    data[['Code1', 'Code2', 'Code3']] = data['საქმიანობის კოდი NACE Rev.2'].str.split('.', expand=True)
    data.drop(columns=['Code2', 'Code3'], inplace=True)

    # Merge with NACE Rev.2 descriptions
    nace_rev2['Rev 2.1 code'] = nace_rev2['Rev 2.1 code'].astype(str)
    data = data.merge(
        nace_rev2,
        left_on='Code1',
        right_on='Rev 2.1 code',
        how='left'
    )

    # Keep relevant columns
    data = data[['საიდენტიფიკაციო ნომერი', 'Name', 'Lat', 'Long']].dropna()

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data['Lat'], data['Long']),
        crs="EPSG:4326"
    )
    return gdf


gdf = load_data()


# ======================
# 2. Helper functions
# ======================
def get_neighborhood_profile(store_id, gdf, radius=30):
    store = gdf[gdf['საიდენტიფიკაციო ნომერი'] == store_id].iloc[0]
    center = store.geometry

    # Convert radius from meters to approximate degrees
    radius_degrees = radius / 111000

    buffer = center.buffer(radius_degrees)
    nearby = gdf[gdf.geometry.within(buffer)]
    composition = nearby['Name'].value_counts(normalize=True)
    return composition, buffer


def get_stores_outside_neighborhood(target_store_id, gdf, radius=30):
    """Get all stores outside the target store's neighborhood"""
    target_store = gdf[gdf['საიდენტიფიკაციო ნომერი'] == target_store_id].iloc[0]
    center = target_store.geometry

    # Convert radius from meters to approximate degrees
    radius_degrees = radius / 111000

    # Create buffer around target store
    target_buffer = center.buffer(radius_degrees)

    # Get stores outside this buffer
    stores_outside = gdf[~gdf.geometry.within(target_buffer)]
    return stores_outside


@st.cache_data
def compute_profiles(_gdf, radius=30):
    profiles = []
    store_ids = _gdf['საიდენტიფიკაციო ნომერი'].tolist()

    for sid in store_ids:
        comp, _ = get_neighborhood_profile(sid, _gdf, radius)
        comp.name = sid
        profiles.append(comp)

    # Create DataFrame with all possible categories
    all_categories = _gdf['Name'].unique()
    profiles_df = pd.DataFrame(profiles).fillna(0)

    # Ensure all categories are present (add missing columns with zeros)
    for category in all_categories:
        if category not in profiles_df.columns:
            profiles_df[category] = 0

    profiles_df = profiles_df.reindex(columns=all_categories, fill_value=0)
    profiles_df.index = store_ids
    profiles_df.index.name = 'store_id'

    return profiles_df


def find_most_similar_neighborhood(target_id, gdf, profiles_df, radius=30):
    """Find the most similar neighborhood outside the target store's radius"""

    # Get all stores outside the target store's neighborhood
    stores_outside = get_stores_outside_neighborhood(target_id, gdf, radius)

    if len(stores_outside) == 0:
        st.warning("No stores found outside the target neighborhood. Try reducing the radius.")
        return None, 0, {}

    # Get the target store's profile
    target_profile = profiles_df.loc[target_id]

    # Calculate similarity with all stores outside the neighborhood
    similarities = {}
    for store_id in stores_outside['საიდენტიფიკაციო ნომერი']:
        if store_id != target_id:  # Double check
            other_profile = profiles_df.loc[store_id]
            similarity = cosine_similarity([target_profile], [other_profile])[0][0]
            similarities[store_id] = similarity

    if not similarities:
        st.warning("No similar stores found outside the target neighborhood.")
        return None, 0, {}

    # Find the most similar store
    most_similar_id = max(similarities, key=similarities.get)
    similarity_score = similarities[most_similar_id]

    return most_similar_id, similarity_score, similarities


def folium_static(map_object, width=700, height=500):
    map_html = map_object.get_root().render()
    html(map_html, width=width, height=height)


# ======================
# 3. Streamlit UI
# ======================
st.title("🗺️ Neighborhood Similarity Explorer")

radius = st.slider("Neighborhood radius (meters):", 10, 100, 30, step=5)

store_options = gdf['საიდენტიფიკაციო ნომერი'].astype(str)
store_choice = st.selectbox("Select a store:", store_options)

target_id = int(store_choice.strip())
st.write(f"**Selected store ID:** {target_id}")

profiles_df = compute_profiles(gdf, radius)
most_similar_id, similarity_score, all_similarities = find_most_similar_neighborhood(target_id, gdf, profiles_df,
                                                                                     radius)

if most_similar_id is None:
    st.stop()

# ======================
# 4. Map visualization
# ======================
target_store = gdf[gdf['საიდენტიფიკაციო ნომერი'] == target_id].iloc[0]
similar_store = gdf[gdf['საიდენტიფიკაციო ნომერი'] == most_similar_id].iloc[0]

# Get target neighborhood boundary for visualization
_, target_buffer = get_neighborhood_profile(target_id, gdf, radius)

# Create map
m = folium.Map(
    location=[gdf['Long'].mean(), gdf['Lat'].mean()],
    tiles='CartoDB positron',
    zoom_start=14.5,
)

# Add target neighborhood boundary
target_buffer_gdf = gpd.GeoDataFrame([1], geometry=[target_buffer], crs=gdf.crs).to_crs(epsg=4326)
folium.GeoJson(
    target_buffer_gdf.__geo_interface__,
    style_function=lambda x: {'fillColor': 'red', 'color': 'red', 'weight': 2, 'fillOpacity': 0.1}
).add_to(m)

# Add all stores as background context
for idx, row in gdf.iterrows():
    if row['საიდენტიფიკაციო ნომერი'] == target_id:
        # Target store - use red marker
        folium.Marker(
            location=[row['Long'], row['Lat']],
            popup=f"ID: {row['საიდენტიფიკაციო ნომერი']}<br>Name: {row['Name']}",
            tooltip=f"Target - ID: {row['საიდენტიფიკაციო ნომერი']}, {row['Name']}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    elif row['საიდენტიფიკაციო ნომერი'] == most_similar_id:  # FIXED: removed ,'Name'
        # Most similar store - use blue marker
        folium.Marker(
            location=[row['Long'], row['Lat']],
            popup=f"ID: {row['საიდენტიფიკაციო ნომერი']}<br>Name: {row['Name']}",
            tooltip=f"Similar - ID: {row['საიდენტიფიკაციო ნომერი']}, {row['Name']}",  # FIXED: Changed to "Similar"
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
    else:
        # Other stores - use gray circle marker
        folium.CircleMarker(
            location=[row['Long'], row['Lat']],
            radius=3,
            color='gray',
            fill=True,
            fill_color='gray',
            fill_opacity=0.3,
            popup=f"ID: {row['საიდენტიფიკაციო ნომერი']}<br>Name: {row['Name']}",
            tooltip=f"ID: {row['საიდენტიფიკაციო ნომერი']}, {row['Name']}",  # FIXED: Removed "Target -"
        ).add_to(m)

st.subheader("Map of Target and Most Similar Neighborhood")
st.write("Red circle: Target neighborhood boundary | Red dot: Target store | Blue dot: Most similar store")
folium_static(m, width=750, height=500)

# ======================
# 5. Composition comparison plot
# ======================
st.subheader("Neighborhood Composition Comparison")

target_profile, _ = get_neighborhood_profile(target_id, gdf, radius=radius)
similar_profile, _ = get_neighborhood_profile(most_similar_id, gdf, radius=radius)

# Create comparison DataFrame with consistent categories
all_categories = list(set(target_profile.index) | set(similar_profile.index))
comparison_data = {
    'Target Store': [target_profile.get(cat, 0) for cat in all_categories],
    'Most Similar Store': [similar_profile.get(cat, 0) for cat in all_categories]
}
comparison = pd.DataFrame(comparison_data, index=all_categories)

fig, ax = plt.subplots(figsize=(10, 5))
comparison.plot.bar(ax=ax)
ax.set_title(f"Neighborhood Composition Comparison\nSimilarity Score: {similarity_score:.4f}")
ax.set_ylabel("Proportion")
ax.set_xlabel("Business Categories")
ax.tick_params(axis='x', rotation=90, labelsize=8)
plt.tight_layout()
st.pyplot(fig)

# Display store information
st.subheader("Store Information")
col1, col2 = st.columns(2)

with col1:
    st.write("**Target Store:**")
    st.write(f"ID: {target_id}")
    st.write(f"Name: {target_store['Name']}")
    st.write(f"Location: (Lat: {target_store['Long']:.6f}, Long: {target_store['Lat']:.6f})")
    st.write(f"Stores in neighborhood: {len(get_neighborhood_profile(target_id, gdf, radius)[0])}")

with col2:
    st.write("**Most Similar Store:**")
    st.write(f"ID: {most_similar_id}")
    st.write(f"Name: {similar_store['Name']}")
    st.write(f"Location: (Lat: {similar_store['Long']:.6f}, Long: {similar_store['Lat']:.6f})")
    st.write(f"Similarity Score: {similarity_score:.4f}")
    st.write(f"Distance from target: {target_store.geometry.distance(similar_store.geometry) * 111000:.1f} meters")

# Debug information - FIXED: Now using the same similarity calculations
with st.expander("Debug Information"):
    stores_outside = get_stores_outside_neighborhood(target_id, gdf, radius)
    st.write(f"Total stores in dataset: {len(gdf)}")
    st.write(f"Stores in target neighborhood: {len(gdf) - len(stores_outside)}")
    st.write(f"Stores outside target neighborhood: {len(stores_outside)}")

    # Show top 10 similar stores using the SAME similarity calculations
    st.write("Top 10 similar stores outside neighborhood:")

    # Get top similar stores from the pre-calculated similarities
    top_similar_stores = sorted(all_similarities.items(), key=lambda x: x[1], reverse=True)[:10]

    similarity_data = []
    for store_id, similarity in top_similar_stores:
        store_name = gdf[gdf['საიდენტიფიკაციო ნომერი'] == store_id].iloc[0]['Name']
        distance = target_store.geometry.distance(
            gdf[gdf['საიდენტიფიკაციო ნომერი'] == store_id].iloc[0].geometry
        ) * 111000

        similarity_data.append({
            'Store ID': store_id,
            'Name': store_name,
            'Similarity': f"{similarity:.4f}",
            'Distance (m)': f"{distance:.1f}"
        })

    debug_df = pd.DataFrame(similarity_data)
    st.dataframe(debug_df)

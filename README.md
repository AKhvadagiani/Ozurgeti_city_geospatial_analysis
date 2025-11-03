**Business Location Intelligence System**

This repository contains a Python pipeline for analyzing business
locations in Ozurgeti municipality, Georgia. The system processes raw business data,
standardizes addresses, geocodes locations, and provides neighborhood similarity analysis
through an interactive web application.

Project Structure

├── data_preparation.py         # Address standardization for Ozurgeti municipality
├── ozurgeti_coordinates.py     # Geocoding and spatial filtering
├── spatial_analysis.py         # Interactive web application for neighborhood analysis
├── main.py                     # Main orchestration script (optional)
└── data/
    ├── ozurgeti.xlsx           # Raw business data
    ├── ozurgeti_streets.txt    # Reference street names
    ├── NACE Rev2.xlsx          # Industry classification codes
    └── geoBoundaries-GEO-ADM2.geojson  # Geographic boundaries

1. Address Clean & Match  (data_preparation.py)
Purpose: Standardizes and cleans business addresses in Ozurgeti municipality using fuzzy matching algorithms.

Key Features:
Extracts and cleans street names from Georgian address strings;
Matches addresses against reference street database;
Uses RapidFuzz for fuzzy string matching;
Generates standardized address format.

2. Geocoding & Spatial Filtering (ozurgeti_coordinates.py)
Purpose: Converts standardized addresses to geographic coordinates and filters points within Ozurgeti municipality boundaries.

Key Features:
Google Maps API integration for geocoding;
Spatial filtering using municipality boundaries;
GeoJSON boundary extraction and visualization;
Rate-limited API calls to avoid limits;

Dependencies:
geopandas, shapely, googlemaps, matplotlib

3. Neighborhood Similarity Analysis (spatial_analysis.py)

Purpose: Interactive web application for exploring neighborhood similarities and business composition.

Key Features:
Interactive map visualization with Folium;
Neighborhood similarity analysis using cosine similarity;
Real-time radius adjustment for neighborhood definition;
Business composition comparison charts;
Detailed store information and metrics.

Launch Application:
streamlit run streamlit_app.py
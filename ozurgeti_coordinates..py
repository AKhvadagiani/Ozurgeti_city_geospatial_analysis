import time
import pandas as pd
import geopandas as gpd
import googlemaps
from shapely.geometry import Point
from matplotlib import pyplot as plt


INPUT_FILE = "ozurgeti_street_with_fuzzy.xlsx"
OUTPUT_FILE = "ozurgeti_dataset_coordinates_filtered.xlsx"
BOUNDARY_FILE = "geoBoundaries-GEO-ADM2.geojson"
OZURGETI_FILE = "ozurgeti_municipality.geojson"
API_KEY = "***"


def load_data(filepath: str) -> pd.DataFrame:
    """Load the Excel dataset."""
    return pd.read_excel(filepath)


def initialize_gmaps(api_key: str):
    """Initialize Google Maps API client."""
    return googlemaps.Client(key=api_key)


def geocode_addresses(gmaps_client, addresses: list, delay: float = 0.2) -> list:
    """
    Geocode a list of addresses and return [(address, lat, lng)].
    Adds a small delay to avoid hitting API rate limits.
    """
    results = []
    for address in addresses:
        if not isinstance(address, str) or not address.strip():
            continue
        try:
            geocode_result = gmaps_client.geocode(address)
            if geocode_result:
                loc = geocode_result[0]["geometry"]["location"]
                results.append((address, loc["lat"], loc["lng"]))
        except Exception as e:
            print(f"Error geocoding {address}: {e}")
        time.sleep(delay)
    return results


def match_coordinates(row, geocoded_list):
    """Find matching geocode for a street name."""
    for entry in geocoded_list:
        if isinstance(row["St_Full_Name"], str) and row["St_Full_Name"].strip() in entry[0]:
            return entry
    return None


def extract_ozurgeti_boundary(geojson_path: str, output_path: str):
    """Extract Ozurgeti municipality boundary from ADM2 dataset."""
    gdf = gpd.read_file(geojson_path)
    oz = gdf[gdf["shapeName"].str.contains("Ozurgeti", case=False)]
    oz.to_file(output_path, driver="GeoJSON")
    return oz


def plot_boundary(gdf):
    """Plot the Ozurgeti boundary."""
    gdf.plot(figsize=(10, 10), color="lightblue", edgecolor="black")
    plt.title("Ozurgeti Municipality Boundary")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid()
    plt.show()


def filter_points_within_boundary(data: pd.DataFrame, boundary_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Filter points that fall within the Ozurgeti boundary."""
    polygon = boundary_gdf.geometry.iloc[0]

    gdf_points = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.Long, data.Lat),
        crs="EPSG:4326"
    ).to_crs(boundary_gdf.crs)

    mask = gdf_points.geometry.within(polygon)
    filtered = data[mask].copy()
    print(f"Filtered {len(filtered)}/{len(data)} points inside Ozurgeti.")
    return filtered


def main():
    data = load_data(INPUT_FILE)

    gmaps_client = initialize_gmaps(API_KEY)

    addresses = [i for i in data["St_Full_Name"] if isinstance(i, str) and i.strip()]
    geocoded = geocode_addresses(gmaps_client, addresses)

    data["კოორდინატები"] = data.apply(lambda row: match_coordinates(row, geocoded), axis=1)
    data[["St_Full_Name_2", "Lat", "Long"]] = data["კოორდინატები"].apply(pd.Series)
    data.drop(columns="St_Full_Name_2", inplace=True)

    oz_gdf = extract_ozurgeti_boundary(BOUNDARY_FILE, OZURGETI_FILE)
    plot_boundary(oz_gdf)

    filtered_data = filter_points_within_boundary(data, oz_gdf)

    filtered_data.to_excel(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()

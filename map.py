#!/usr/bin/env python3
import json
import pandas as pd
import folium
import geopandas as gpd
from shapely.geometry import Point
from folium.plugins import MarkerCluster

SF_GEO_PATH = "SF.geojson"
JSONL_PATH = "detections.jsonl"
OUT_HTML   = "index.html"

# Optional: map class_id -> class_name if you use numeric IDs in logs
ID_TO_NAME = {0: "tree", 1: "bush", 2: "grass"}

hoods = gpd.read_file(SF_GEO_PATH)

# Auto-detect the neighborhood name column in your GeoJSON
NEIGH_COL = next(c for c in ["name", "neighborhood", "nhood", "label"] if c in hoods.columns)

def tag_points_with_hoods(df, lat_col="latitude", lon_col="longitude"):
    gdf_points = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(gdf_points, hoods[[NEIGH_COL, "geometry"]], predicate="within", how="left")
    joined = joined.rename(columns={NEIGH_COL: "neighborhood"}).drop(columns=["index_right"])
    return joined

# San Francisco center & a loose bounding box (optional filter)
SF_CENTER = (37.7749, -122.4194)
SF_BBOX   = {
    "min_lat": 37.5,
    "max_lat": 37.90,
    "min_lon": -122.55,
    "max_lon": -122.30
}

def load_records(jsonl_path):
    rows = []
    with open(jsonl_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            lat = r.get("latitude")
            lon = r.get("longitude")
            if lat is None or lon is None:
                continue  # skip entries without GPS

            # Resolve class name
            cid = r.get("class_id")
            cname = ID_TO_NAME.get(cid, f"class_{cid}")

            # Optional: only keep points roughly around SF
            if (SF_BBOX["min_lat"] <= lat <= SF_BBOX["max_lat"] and
                SF_BBOX["min_lon"] <= lon <= SF_BBOX["max_lon"]):
                rows.append({
                    "track_id":  r.get("track_id"),
                    "class_id":  r.get("class_id"),
                    "class_name": cname,
                    "confidence": r.get("confidence"),
                    "latitude":  lat,
                    "longitude": lon,
                })
    return pd.DataFrame(rows)

def main():
    df = load_records(JSONL_PATH)
    if df.empty:
        print("No valid GPS detections found in", JSONL_PATH)
        return

    df = df.drop_duplicates(subset=["track_id", "class_name", "latitude", "longitude"])

    # Base map (create once)
    m = folium.Map(location=SF_CENTER, zoom_start=12, tiles="cartodbpositron")

    # Neighborhood tagging
    joined = tag_points_with_hoods(df, lat_col="latitude", lon_col="longitude")

    # Totals per neighborhood for choropleth
    total_counts = (
        joined.groupby("neighborhood")
        .size()
        .reset_index(name="total_count")
    )
    hoods_total = hoods.merge(
        total_counts,
        left_on=NEIGH_COL,
        right_on="neighborhood",
        how="left"
    )
    hoods_total["total_count"] = hoods_total["total_count"].fillna(0)

    # Choropleth layer (density)
    choropleth = folium.Choropleth(
        geo_data=hoods_total,
        data=hoods_total,
        columns=["neighborhood", "total_count"],
        key_on=f"feature.properties.{NEIGH_COL}",
        fill_color="YlGn",
        fill_opacity=0.75,
        line_opacity=0.3,
        nan_fill_opacity=0.1,
        legend_name="Total Foliage Detections",
        name="Foliage Density"
    )
    choropleth.add_to(m)

    folium.GeoJsonTooltip(
        fields=[NEIGH_COL, "total_count"],
        aliases=["Neighborhood", "Detections"],
        localize=True,
        sticky=False
    ).add_to(choropleth.geojson)

    # --- POINT LAYERS ---
    # Color mapping by class
    color_for = {"tree": "brown", "bush": "green", "grass": "purple"}
    def color_of(c): return color_for.get(str(c).lower(), "gray")


    # Individual points (OFF by default)
    points_individual = folium.FeatureGroup(name="Individual Points", show=False)

    for _, row in joined.iterrows():
        lat, lon = float(row["latitude"]), float(row["longitude"])
        popup = folium.Popup(
            f"<b>Class:</b> {row.get('class_name','?')}<br>"
            f"<b>Neighborhood:</b> {row.get('neighborhood','(none)')}",
            max_width=250
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            weight=1,
            fill=True,
            fill_opacity=0.8,
            color=color_of(row.get("class_name")),
            popup=popup
        ).add_to(points_individual)

    points_individual.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = """
    <div style="
        position: fixed; 
        bottom: 30px; left: 30px; width: 150px; 
        background-color: white; 
        border:2px solid grey; 
        z-index:9999; 
        font-size:14px;
        padding: 8px;
    ">
    <b>Foliage Legend</b><br>
    <i style="background:brown; width:12px; height:12px; float:left; margin-right:6px; opacity:0.8"></i> Tree<br>
    <i style="background:green; width:12px; height:12px; float:left; margin-right:6px; opacity:0.8"></i> Bush<br>
    <i style="background:purple; width:12px; height:12px; float:left; margin-right:6px; opacity:0.8"></i> Grass
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(OUT_HTML)
    print("Wrote", OUT_HTML)

if __name__ == "__main__":
    main()

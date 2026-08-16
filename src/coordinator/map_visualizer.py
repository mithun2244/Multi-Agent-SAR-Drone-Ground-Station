"""Drop the demo case target on an interactive map and open it in a browser."""

import os
import webbrowser

import folium

# The demo case coordinates (see coordinator/demo.py).
TARGET_LAT, TARGET_LON = 46.83160, 8.22760
OUTPUT = os.path.join(os.getcwd(), "sar_target_map.html")


def build_map(lat=TARGET_LAT, lon=TARGET_LON, path=OUTPUT):
    m = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker(
        [lat, lon],
        popup="SAR Target Detected - Priority Rescue",
        icon=folium.Icon(color="red"),
    ).add_to(m)
    m.save(path)
    return path


if __name__ == "__main__":
    path = build_map()
    print(f"Map written to {path}")
    webbrowser.open(f"file://{path}")

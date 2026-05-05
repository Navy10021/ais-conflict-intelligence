"""
Conflict Overlay Visualization
=======================
Overlay conflict events on maps.
"""
import pandas as pd
from pathlib import Path
import logging
import argparse

logger = logging.getLogger(__name__)


class ConflictOverlayVisualizer:
    """Overlay conflict events on spatial visualizations."""

    def __init__(self, output_dir: str = "outputs/figures/spatial"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_conflict_events(
        self, df: pd.DataFrame, events_path: str
    ) -> str:
        """
        Plot conflict events on a Folium map with AIS density background.
        """
        try:
            import folium
            from folium.plugins import HeatMap
        except ImportError:
            logger.error("folium not installed. Skipping conflict overlay.")
            return ""

        logger.info("Creating conflict overlay map...")
        events = pd.read_csv(events_path, parse_dates=["event_date"])

        center_lat = df["lat"].mean()
        center_lon = df["lon"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

        # AIS density heatmap
        heat_data = [[row.lat, row.lon] for _, row in df.dropna(subset=["lat", "lon"]).iterrows()]
        HeatMap(heat_data, radius=10, blur=5, max_zoom=10).add_to(m)

        # Conflict events as markers
        for _, ev in events.iterrows():
            folium.Marker(
                location=[ev.get("lat", center_lat), ev.get("lon", center_lon)],
                popup=f"Date: {ev['event_date'].date()}<br>Type: {ev.get('event_type', 'N/A')}",
                icon=folium.Icon(color="red", icon="warning-sign"),
            ).add_to(m)

        output_path = self.output_dir / "conflict_overlay.html"
        m.save(str(output_path))
        logger.info("Saved conflict overlay to {output_path}")
        return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Overlay conflict events on maps")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--events", required=True, help="Conflict events CSV")
    parser.add_argument("--output-dir", default="outputs/figures/spatial")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_parquet(args.input)
    viz = ConflictOverlayVisualizer(output_dir=args.output_dir)
    viz.plot_conflict_events(df, args.events)

    print(f"Conflict overlay saved to {args.output_dir}")


if __name__ == "__main__":
    main()


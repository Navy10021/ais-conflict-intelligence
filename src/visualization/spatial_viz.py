"""
Spatial Visualization Module
=======================
Outputs ??outputs/figures/spatial/

Figures:
  1. Global density heatmap ??Folium HeatMapWithTime, 6-hour steps
  2. Per-MMSI trajectory ??SOG color-mapped polyline
  3. Dark ship cluster map ??DBSCAN inside conflict zones
  4. Destination flow map ??origin ??destination Sankey/arc map
  5. AtoN / Base Station coverage map ??aisdk receiver footprint
"""
import pandas as pd
from pathlib import Path
import logging
import argparse
from typing import Optional

logger = logging.getLogger(__name__)


class SpatialVisualizer:
    """Generate spatial visualizations for AIS data."""

    def __init__(self, output_dir: str = "outputs/figures/spatial"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_density_heatmap(
        self, df: pd.DataFrame, title: str = "AIS Density Heatmap"
    ) -> str:
        """
        Create Folium HeatMapWithTime showing vessel density.
        Groups by 6-hour time buckets.
        """
        try:
            import folium
            from folium.plugins import HeatMapWithTime
        except ImportError:
            logger.error("folium not installed. Skipping heatmap.")
            return ""

        logger.info("Creating density heatmap...")
        df = df.dropna(subset=["lat", "lon", "timestamp"])
        df["time_bucket"] = df["timestamp"].dt.floor("6h").astype(str)

        heat_data = []
        for t_bucket in sorted(df["time_bucket"].unique()):
            sub = df[df["time_bucket"] == t_bucket]
            heat_data.append([[row.lat, row.lon] for _, row in sub.iterrows()])

        center_lat = df["lat"].mean()
        center_lon = df["lon"].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
        hm = HeatMapWithTime(
            heat_data,
            index=list(sorted(df["time_bucket"].unique())),
            radius=15,
            blur=10,
            max_opacity=0.8,
        )
        hm.add_to(m)

        output_path = self.output_dir / "density_heatmap.html"
        m.save(str(output_path))
        logger.info("Saved heatmap to {output_path}")
        return str(output_path)

    def plot_trajectories(
        self, df: pd.DataFrame, mmsi_list: Optional[list] = None, max_vessels: int = 5
    ) -> str:
        """
        Plot per-MMSI trajectories with SOG color mapping.
        """
        try:
            import folium
            from folium.plugins import AntPath
        except ImportError:
            logger.error("folium not installed. Skipping trajectory plot.")
            return ""

        logger.info("Creating trajectory plot...")
        df = df.dropna(subset=["lat", "lon", "mmsi"]).sort_values(["mmsi", "timestamp"])

        if mmsi_list is None:
            mmsi_list = df["mmsi"].unique()[:max_vessels]

        center_lat = df["lat"].mean()
        center_lon = df["lon"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

        for mmsi in mmsi_list:
            vessel_data = df[df["mmsi"] == mmsi]
            points = [[row.lat, row.lon] for _, row in vessel_data.iterrows()]

            if len(points) < 2:
                continue

            color = "blue"
            if "sog" in vessel_data.columns:
                avg_sog = vessel_data["sog"].mean()
                if avg_sog > 15:
                    color = "red"
                elif avg_sog > 8:
                    color = "orange"
                else:
                    color = "green"

            AntPath(
                locations=points,
                color=color,
                weight=2,
                opacity=0.7,
                pulse_pattern=dict(dash=10, gap=5),
            ).add_to(m)

        output_path = self.output_dir / "trajectories.html"
        m.save(str(output_path))
        logger.info("Saved trajectories to {output_path}")
        return str(output_path)

    def plot_dark_ship_clusters(
        self, df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5
    ) -> str:
        """
        Cluster dark ships using DBSCAN and plot on map.
        """
        try:
            import folium
            from sklearn.cluster import DBSCAN
        except ImportError:
            logger.error("Required libraries not installed. Skipping cluster plot.")
            return ""

        logger.info("Clustering dark ships...")
        dark_ships = df[df.get("is_dark_ship", 0) == 1].dropna(subset=["lat", "lon"])

        if len(dark_ships) < min_samples:
            logger.warning("Not enough dark ship data for clustering.")
            return ""

        coords = dark_ships[["lat", "lon"]].values
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        clusters = dbscan.fit_predict(coords)
        dark_ships = dark_ships.copy()
        dark_ships["cluster"] = clusters

        center_lat = dark_ships["lat"].mean()
        center_lon = dark_ships["lon"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=8)

        colors = ["red", "blue", "green", "purple", "orange", "yellow", "pink"]
        for cluster_id in dark_ships["cluster"].unique():
            cluster_data = dark_ships[dark_ships["cluster"] == cluster_id]
            color = colors[int(cluster_id) % len(colors)] if cluster_id != -1 else "black"

            for _, row in cluster_data.iterrows():
                folium.CircleMarker(
                    location=[row.lat, row.lon],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=f"MMSI: {row.mmsi}",
                ).add_to(m)

        output_path = self.output_dir / "dark_ship_clusters.html"
        m.save(str(output_path))
        logger.info("Saved dark ship clusters to {output_path}")
        return str(output_path)

    def plot_ais_coverage(self, df: pd.DataFrame) -> str:
        """
        Plot AIS receiver coverage based on data density.
        """
        try:
            import folium
        except ImportError:
            logger.error("folium not installed. Skipping coverage plot.")
            return ""

        logger.info("Creating AIS coverage map...")
        center_lat = df["lat"].mean()
        center_lon = df["lon"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

        from folium.plugins import HeatMap
        heat_data = [[row.lat, row.lon, 1] for _, row in df.dropna(subset=["lat", "lon"]).iterrows()]
        HeatMap(heat_data, radius=10, blur=5, max_zoom=10).add_to(m)

        output_path = self.output_dir / "ais_coverage.html"
        m.save(str(output_path))
        logger.info("Saved coverage map to {output_path}")
        return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate spatial visualizations")
    parser.add_argument("--input", required=True, help="Input parquet file (features)")
    parser.add_argument("--output-dir", default="outputs/figures/spatial",
                        help="Output directory for figures")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded {len(df):,} records from {args.input}")

    viz = SpatialVisualizer(output_dir=args.output_dir)

    viz.plot_density_heatmap(df)
    viz.plot_trajectories(df, max_vessels=5)
    viz.plot_dark_ship_clusters(df)
    viz.plot_ais_coverage(df)

    print(f"Spatial visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()


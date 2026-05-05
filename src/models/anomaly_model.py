"""
Unsupervised Maritime Anomaly Detection
=======================================
Models:
  1. Isolation Forest       ??multivariate point anomaly score
  2. Variational Autoencoder ??reconstruction error on normal pattern
  3. DBSCAN                  ??spatial density cluster anomaly
  4. Local Outlier Factor    ??neighborhood density deviation

Anomaly type taxonomy:
  "dark_ship"         ??is_dark_ship == 1 (AIS gap > 6h)
  "evasive_maneuver"  ??high ROT + high SOG in conflict zone (NEW)
  "loitering"         ??slow + frequent turns in conflict zone
  "zig_zag"           ??high zig_zag_index
  "density_surge"     ??traffic_count spike vs. rolling baseline
  "destination_spoof" ??eta_implausible or dest_changed frequently (NEW)
  "speed_spike"       ??sog_z_score > 3?
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from typing import Optional, Any

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """Isolation Forest for multivariate anomaly detection."""

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model: Optional[Any] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict anomalies."""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.error("scikit-learn not installed.")
            return np.zeros(len(X))

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        self.model.fit(X)
        return self.model.predict(X)  # 1 = normal, -1 = anomaly


class VAEDetector:
    """Variational Autoencoder for anomaly detection."""

    def __init__(self, input_dim: int, latent_dim: int = 10, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.model: Optional[Any] = None
        self._build_model()

    def _build_model(self):
        """Build VAE model."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.error("PyTorch not installed.")
            return

        class VAE(nn.Module):
            def __init__(self, input_dim, latent_dim, hidden_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                )
                self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
                self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                    nn.Sigmoid(),
                )

            def encode(self, x):
                h = self.encoder(x)
                return self.fc_mu(h), self.fc_logvar(h)

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                return self.decoder(z)

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar

        self.model = VAE(self.input_dim, self.latent_dim, self.hidden_dim)

    def fit(self, X: np.ndarray, epochs: int = 50, lr: float = 0.001):
        """Train VAE."""
        if self.model is None:
            return

        try:
            import torch
            import torch.nn as nn
            from torch.optim import Adam

            X_tensor = torch.FloatTensor(X)
            optimizer = Adam(self.model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                recon, mu, logvar = self.model(X_tensor)
                # VAE loss: reconstruction + KL divergence
                recon_loss = criterion(recon, X_tensor)
                kl_loss = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp()
                )
                loss = recon_loss + kl_loss
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:
                    logger.info("Epoch {epoch}, Loss: {loss.item():.4f}")
        except ImportError:
            logger.error("PyTorch not installed.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction error."""
        if self.model is None:
            return np.zeros(len(X))

        try:
            import torch
            import torch.nn as nn

            self.model.eval()
            X_tensor = torch.FloatTensor(X)
            with torch.no_grad():
                recon, _, _ = self.model(X_tensor)
                error = nn.MSELoss(reduction="none")(recon, X_tensor).mean(dim=1)
            return error.numpy()
        except ImportError:
            logger.error("PyTorch not installed.")
            return np.zeros(len(X))


class DBSCANDetector:
    """DBSCAN for spatial density anomaly detection."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples

    def detect(self, coords: np.ndarray) -> np.ndarray:
        """Detect spatial anomalies."""
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            logger.error("scikit-learn not installed.")
            return np.zeros(len(coords))

        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        clusters = dbscan.fit_predict(coords)
        return clusters  # -1 = outlier


class AnomalyModel:
    """Combined anomaly detection framework."""

    ANOMALY_TYPES = [
        "dark_ship", "evasive_maneuver", "loitering", "zig_zag",
        "density_surge", "destination_spoof", "speed_spike",
    ]

    def __init__(self, output_dir: str = "outputs/models/anomaly"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[dict[str, Any]] = []

    def detect_by_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using rule-based classification."""
        logger.info("Running rule-based anomaly detection...")
        df = df.copy()

        # Initialize anomaly columns
        for atype in self.ANOMALY_TYPES:
            df[f"is_{atype}"] = 0

        # Dark ship
        if "is_dark_ship" in df.columns:
            df["is_dark_ship"] = df["is_dark_ship"]

        # Evasive maneuver
        if all(c in df.columns for c in ["rot_abs", "sog", "in_conflict_zone"]):
            df["is_evasive_maneuver"] = (
                (df["rot_abs"] > 10.0) & (df["sog"] > 5.0) & df["in_conflict_zone"]
            ).astype(int)

        # Loitering
        if all(c in df.columns for c in ["sog", "delta_cog", "in_conflict_zone"]):
            df["is_loitering"] = (
                (df["sog"] < 3.0) & (df["delta_cog"] > 45.0) & df["in_conflict_zone"]
            ).astype(int)

        # Zig-zag
        if "zig_zag_index" in df.columns:
            df["is_zig_zag"] = (df["zig_zag_index"] > 5).astype(int)

        # Speed spike
        if "sog_z_score" in df.columns:
            df["is_speed_spike"] = (df["sog_z_score"].abs() > 3).astype(int)

        # Destination spoof
        if "eta_implausible" in df.columns:
            df["is_destination_spoof"] = df["eta_implausible"]

        # Density surge (requires temporal aggregation)
        if "traffic_count" in df.columns:
            df["traffic_rolling"] = df.groupby("grid_cell")["traffic_count"].transform(
                lambda x: x.rolling("24h", min_periods=3).mean()
            )
            threshold = 2 * df["traffic_rolling"]
            df["is_density_surge"] = (
                df["traffic_count"] > threshold
            ).astype(int)

        # Summary
        anomaly_cols = [
            f"is_{a}" for a in self.ANOMALY_TYPES
            if f"is_{a}" in df.columns
        ]
        df["any_anomaly"] = df[anomaly_cols].any(axis=1).astype(int)

        count = df['any_anomaly'].sum()
        logger.info("Detected {count:,} anomaly records")
        return df

    def detect_multivariate(self, df: pd.DataFrame, features: Optional[list] = None) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest."""
        logger.info("Running Isolation Forest...")
        df = df.copy()

        if features is None:
            features = ["sog", "cog", "rot", "rot_abs", "delta_sog", "delta_cog"]

        available = [f for f in features if f in df.columns]
        if len(available) < 2:
            logger.warning("Not enough features for multivariate detection.")
            return df

        X = df[available].fillna(df[available].median()).values

        detector = IsolationForestDetector()
        predictions = detector.fit_predict(X)
        df["is_anomaly_mv"] = (predictions == -1).astype(int)

        logger.info("Multivariate anomalies: {df['is_anomaly_mv'].sum():,}")
        return df

    def run_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all anomaly detection methods."""
        df = self.detect_by_rules(df)
        df = self.detect_multivariate(df)

        # Save results
        output_path = self.output_dir / "anomaly_detection_results.parquet"
        df.to_parquet(output_path, index=False, compression="snappy")
        logger.info("Saved anomaly results to {output_path}")

        # Summary table
        summary = df[["mmsi", "timestamp"] + [c for c in df.columns if c.startswith("is_")]].copy()
        summary_path = self.output_dir / "anomaly_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("Saved summary to {summary_path}")

        return df


def main():
    parser = argparse.ArgumentParser(description="Anomaly detection")
    parser.add_argument("--input", required=True, help="Input parquet file (features)")
    parser.add_argument("--output", default="outputs/models/anomaly")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded {len(df):,} records")

    model = AnomalyModel(output_dir=args.output)
    results = model.run_all(df)

    print(f"Anomaly detection complete. Results: {results['any_anomaly'].sum():,} anomalies")


if __name__ == "__main__":
    main()


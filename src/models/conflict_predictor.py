"""
Conflict Prediction Model
========================
Predicts P(conflict within T+N days) per grid_cell 횞 time bucket.

Architectures:
  1. Bidirectional LSTM + Multi-Head Attention  (primary temporal)
  2. Temporal Fusion Transformer (TFT)          (static + dynamic inputs)
  3. XGBoost                                    (interpretable baseline)

Training:
  Target   : conflict_label (binary); days_to_conflict (regression)
  Horizons : T+3, T+7, T+14, T+30 days
  Imbalance: SMOTE + focal loss (款=2)
  Split    : temporal ??train ??2023-06, val 2023-07??9, test 2023-10+

Evaluation:
  AUROC, AUPRC, F2-Score, Mean Lead Time, False Alarm Rate
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Reproducibility
SEED = 42
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass


class ConflictLSTM(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers,
                                   batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = torch.nn.MultiheadAttention(
            hidden_dim * 2, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm = torch.nn.LayerNorm(hidden_dim * 2)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, 64), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(64, 1), torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _  = self.lstm(x)
        attn, _ = self.attention(out, out, out)
        return self.head(self.norm(out + attn).mean(dim=1)).squeeze(-1)


class FocalBCELoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        bce = torch.nn.functional.binary_cross_entropy(pred, target, reduction="none")
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class ConflictPredictor:
    """
    Conflict prediction framework with multiple model architectures.
    """

    FEATURES_AGGREGATE = [
        "traffic_count", "class_a_ratio", "dark_ship_ratio",
        "military_ratio", "fishing_ratio", "tanker_ratio",
        "sar_count", "mean_sog", "std_sog", "mean_rot_abs",
        "loitering_density", "not_under_command_count",
        "evasive_count", "dest_conflict_count",
        "destination_count_24h", "route_entropy", "zig_zag_index",
    ]

    def __init__(self, output_dir: str = "outputs/models/predictor"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models: dict[str, Any] = {}
        self.results: dict[str, dict[str, float]] = {}

    def prepare_data(
        self, df: pd.DataFrame, target: str = "conflict_label",
        horizon_days: int = 7
    ) -> tuple:
        """Prepare features and target for modeling."""
        available = [f for f in self.FEATURES_AGGREGATE if f in df.columns]
        if len(available) < 5:
            logger.warning("Not enough features for prediction.")
            return None, None, None

        X = df[available].fillna(0)
        y = df[target].fillna(0) if target in df.columns else None

        return X, y, available

    def train_lstm(
        self, X: pd.DataFrame, y: pd.Series,
        input_dim: Optional[int] = None, epochs: int = 100
    ) -> dict:
        """Train LSTM with Attention model."""
        try:
            import torch  # noqa: F401
        except ImportError:
            logger.error("PyTorch not installed.")
            return {}

        if input_dim is None:
            input_dim = X.shape[1]

        logger.info("Training LSTM with input_dim={input_dim}...")

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.values)

        model = ConflictLSTM(input_dim=input_dim)
        criterion = FocalBCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor.unsqueeze(1))  # Add sequence dimension
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                logger.info("Epoch %d, Loss: %.4f", epoch, loss.item())

        self.models["lstm"] = model
        return {"model": "LSTM", "input_dim": input_dim, "epochs": epochs}

    def train_xgboost(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost not installed.")
            return {}

        logger.info("Training XGBoost...")

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(X, y)

        # Feature importance
        importance = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        self.models["xgboost"] = model

        result = {
            "model": "XGBoost",
            "top_features": importance.head(10).to_dict("records"),
        }
        return result

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
        """Evaluate model performance."""
        if model_name not in self.models:
            logger.warning("Model {model_name} not trained.")
            return {}

        model = self.models[model_name]

        try:
            from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score

            if model_name == "xgboost":
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                try:
                    import torch
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_test.values).unsqueeze(1)
                        y_prob = model(X_tensor).numpy()
                        y_pred = (y_prob > 0.5).astype(int)
                except ImportError:
                    logger.error("PyTorch not installed for LSTM evaluation.")
                    return {}

            auroc = roc_auc_score(y_test, y_prob)
            auprc = average_precision_score(y_test, y_prob)
            f2 = fbeta_score(y_test, y_pred, beta=2)

            result = {
                "model": model_name,
                "auroc": auroc,
                "auprc": auprc,
                "f2_score": f2,
            }
            self.results[model_name] = result
            logger.info("{model_name} - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, F2: {f2:.4f}")
            return result

        except ImportError:
            logger.error("sklearn not installed for evaluation.")
            return {}

    def save_results(self) -> str:
        """Save all results to CSV."""
        if not self.results:
            logger.warning("No results to save.")
            return ""

        results_df = pd.DataFrame(self.results.values())
        output_path = self.output_dir / "prediction_results.csv"
        results_df.to_csv(output_path, index=False)
        logger.info("Saved results to {output_path}")
        return str(output_path)

    def run(
        self, df: pd.DataFrame, target: str = "conflict_label",
        horizon_days: int = 7, train_ratio: float = 0.7
    ) -> dict:
        """Run full training and evaluation pipeline."""
        logger.info("Running conflict predictor (T+{horizon_days})...")
        
        X, y, features = self.prepare_data(df, target, horizon_days)
        if X is None:
            return {}
        
        if y is None:
            logger.warning("Target column '{target}' not found. Creating dummy target.")
            y = pd.Series(np.zeros(len(X)), index=X.index)
        
        # Temporal split
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train models
        self.train_xgboost(X_train, y_train)
        self.train_lstm(X_train, y_train, input_dim=len(features))
        
        # Evaluate (only if we have both classes)
        if y_test.nunique() > 1:
            for model_name in self.models.keys():
                self.evaluate_model(X_test, y_test, model_name)
        else:
            logger.warning("Not enough class diversity for evaluation.")
        
        self.save_results()
        return self.results


def main():
    parser = argparse.ArgumentParser(description="Train conflict prediction model")
    parser.add_argument("--input", required=True, help="Input parquet file (features or aggregated)")
    parser.add_argument("--mode", default="train", choices=["train", "eval"], help="Mode")
    parser.add_argument("--target", default="conflict_label", help="Target column")
    parser.add_argument("--horizon", type=int, default=7, help="Prediction horizon (days)")
    parser.add_argument("--output", default="outputs/models/predictor")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded {len(df):,} records")

    predictor = ConflictPredictor(output_dir=args.output)
    results = predictor.run(df, target=args.target, horizon_days=args.horizon)

    print(f"Prediction complete. Models: {list(results.keys())}")


if __name__ == "__main__":
    main()


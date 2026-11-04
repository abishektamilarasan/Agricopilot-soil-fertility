import os
import json
import joblib
from typing import Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "..", "models")

MODEL_PATH = os.path.join(MODELS_DIR, "soil_fertility_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "soil_fertility_scaler.joblib")

FERTILITY_CLASSES = ["Low", "Medium", "High"]


class SoilFertilityPredictor:
    """
    Standalone Soil Fertility Classification Engine
    Classifies soil fertility index (Low, Medium, High) from N, P, K, pH,
    organic carbon, and soil moisture content.
    """
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

    def predict(
        self,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        ph: float,
        organic_carbon: float,
        moisture: float = 20.0
    ) -> Dict[str, Any]:
        """
        Predict soil fertility tier.
        """
        features = [[nitrogen, phosphorus, potassium, ph, organic_carbon, moisture]]
        
        if self.scaler:
            try:
                features = self.scaler.transform(features)
            except Exception:
                pass  # Fall back to unscaled if feature dims differ
        
        pred_idx = int(self.model.predict(features)[0])
        
        # Probabilities
        probs = None
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(features)[0]
            probs = {FERTILITY_CLASSES[i]: round(float(p[i]) * 100, 2) for i in range(min(len(p), len(FERTILITY_CLASSES)))}
        
        fertility_label = FERTILITY_CLASSES[pred_idx] if pred_idx < len(FERTILITY_CLASSES) else str(pred_idx)
        
        return {
            "fertility_tier": fertility_label,
            "tier_index": pred_idx,
            "probabilities": probs,
            "input_summary": {
                "n": nitrogen, "p": phosphorus, "k": potassium,
                "ph": ph, "organic_carbon": organic_carbon, "moisture": moisture
            }
        }


if __name__ == "__main__":
    predictor = SoilFertilityPredictor()
    sample = {
        "nitrogen": 140.0,
        "phosphorus": 35.0,
        "potassium": 210.0,
        "ph": 6.8,
        "organic_carbon": 0.75,
        "moisture": 22.5
    }
    result = predictor.predict(**sample)
    print("Standalone Soil Fertility Prediction:")
    print(json.dumps(result, indent=2))

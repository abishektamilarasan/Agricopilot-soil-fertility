# 🌱 AgriCopilot — Soil Fertility Assessment Model

Standalone machine learning model for soil fertility tier classification developed for [AgriCopilot](https://github.com/abishektamilarasan/Agricopilot). Predicts soil fertility status (`Low`, `Medium`, `High`) using chemical nutrient indicators and physicochemical properties.

---

## 📊 Model Details

- **Algorithm:** Tuned Gradient Boosting / Random Forest Classifier with standard normalization.
- **Classes:** 
  - `0`: Low Fertility
  - `1`: Medium Fertility
  - `2`: High Fertility
- **Input Features (6):**
  - Nitrogen (N) (kg/ha)
  - Phosphorus (P) (kg/ha)
  - Potassium (K) (kg/ha)
  - Soil pH
  - Organic Carbon (OC) (%)
  - Soil Moisture Content (%)
- **Validation Accuracy:** **98.7%**

---

## 📁 Repository Structure

```
.
├── models/
│   ├── soil_fertility_model.joblib   # Trained classifier
│   └── soil_fertility_scaler.joblib  # Feature StandardScaler
├── src/
│   └── predict.py                    # Standalone inference engine
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation & benchmarks
```

---

## 🚀 Quickstart & Standalone Inference

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Inference in Python
```python
from src.predict import SoilFertilityPredictor

predictor = SoilFertilityPredictor()

result = predictor.predict(
    nitrogen=140.0,
    phosphorus=35.0,
    potassium=210.0,
    ph=6.8,
    organic_carbon=0.75,
    moisture=22.5
)

print(result)
```

**Output:**
```json
{
  "fertility_tier": "High",
  "tier_index": 2,
  "probabilities": {
    "Low": 1.25,
    "Medium": 8.45,
    "High": 90.30
  }
}
```

---

## 🔗 Integration with AgriCopilot

The AgriCopilot FastAPI backend consumes this model at:
```http
POST /api/ml/soil/fertility
```
Payload:
```json
{
  "n": 140.0,
  "p": 35.0,
  "k": 210.0,
  "ph": 6.8,
  "organic_carbon": 0.75,
  "moisture": 22.5
}
```

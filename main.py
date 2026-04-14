import joblib
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ── Model Architecture ────────────────────────────────────────
class ShopperMLP(nn.Module):
    def __init__(self, cat_dims, num_numerical, embedding_dim=8,
                 hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=n_cat, embedding_dim=embedding_dim)
            for n_cat in cat_dims
        ])
        total_input = len(cat_dims) * embedding_dim + num_numerical
        layers = []
        in_dim = total_input
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, cat_x, num_x):
        embeddings = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embedded = torch.cat(embeddings, dim=1)
        x = torch.cat([cat_embedded, num_x], dim=1)
        return self.network(x).squeeze(1)

# ── Columns ───────────────────────────────────────────────────
num_cols = ['Administrative', 'Administrative_Duration', 'Informational',
            'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
            'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

cat_cols = ['Month', 'OperatingSystems', 'Browser', 'Region',
            'TrafficType', 'VisitorType', 'Weekend']

# ── Load saved files ──────────────────────────────────────────
scaler         = joblib.load('models/scaler.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
lgb_model      = joblib.load('models/lgb_model.pkl')

# ── Load MLP ──────────────────────────────────────────────────
cat_dims  = [le.classes_.shape[0] + 1 for le in label_encoders.values()]
mlp_model = ShopperMLP(
    cat_dims=cat_dims,
    num_numerical=len(num_cols),
    embedding_dim=8,
    hidden_dims=[256, 128, 64],
    dropout=0.3
)
mlp_model.load_state_dict(torch.load('models/mlp_model.pth', map_location='cpu'))
mlp_model.eval()

print("✅ All models loaded!")

# ── FastAPI ───────────────────────────────────────────────────
app = FastAPI(title="Online Shoppers Prediction API")

# ── Input Schema ──────────────────────────────────────────────
class ShopperInput(BaseModel):
    Administrative: float
    Administrative_Duration: float
    Informational: float
    Informational_Duration: float
    ProductRelated: float
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool

# ── Preprocessing ─────────────────────────────────────────────
def preprocess(data: ShopperInput):
    row = {
        'Administrative':          data.Administrative,
        'Administrative_Duration': data.Administrative_Duration,
        'Informational':           data.Informational,
        'Informational_Duration':  data.Informational_Duration,
        'ProductRelated':          data.ProductRelated,
        'ProductRelated_Duration': data.ProductRelated_Duration,
        'BounceRates':             data.BounceRates,
        'ExitRates':               data.ExitRates,
        'PageValues':              data.PageValues,
        'SpecialDay':              data.SpecialDay,
        'Month':                   data.Month,
        'OperatingSystems':        data.OperatingSystems,
        'Browser':                 data.Browser,
        'Region':                  data.Region,
        'TrafficType':             data.TrafficType,
        'VisitorType':             data.VisitorType,
        'Weekend':                 str(data.Weekend)
    }

    df = pd.DataFrame([row])

    for col in cat_cols:
        le  = label_encoders[col]
        val = df[col].astype(str).values[0]
        df[col] = le.transform([val])[0] if val in le.classes_ else 0

    df[num_cols] = scaler.transform(df[num_cols])
    return df

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Online Shoppers Prediction API is running!"}

@app.post("/predict/mlp")
def predict_mlp(data: ShopperInput):
    try:
        df = preprocess(data)
        cat_tensor = torch.tensor(df[cat_cols].values, dtype=torch.long)
        num_tensor = torch.tensor(df[num_cols].values, dtype=torch.float32)
        with torch.no_grad():
            logit = mlp_model(cat_tensor, num_tensor)
            prob  = torch.sigmoid(logit).item()
        return {"prediction": "yes" if prob >= 0.5 else "no",
                "probability": round(prob, 4)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/tree")
def predict_tree(data: ShopperInput):
    try:
        df   = preprocess(data)
        prob = lgb_model.predict_proba(df[num_cols + cat_cols])[:, 1][0]
        return {"prediction": "yes" if prob >= 0.5 else "no",
                "probability": round(float(prob), 4)}
    except Exception as e:
        return {"error": str(e)}
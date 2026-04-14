# streamlit_app.py
# Online Shoppers Purchase Prediction App

import streamlit as st
import joblib
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# ── Model Architecture (must match training) ──────────────────
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

# ── Load Models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler         = joblib.load('models/scaler.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    lgb_model      = joblib.load('models/lgb_model.pkl')

    cat_dims = [le.classes_.shape[0] + 1 for le in label_encoders.values()]
    mlp_model = ShopperMLP(
        cat_dims=cat_dims,
        num_numerical=len(num_cols),
        embedding_dim=8,
        hidden_dims=[256, 128, 64],
        dropout=0.3
    )
    mlp_model.load_state_dict(torch.load('models/mlp_model.pth',
                                          map_location='cpu'))
    mlp_model.eval()
    return scaler, label_encoders, lgb_model, mlp_model

scaler, label_encoders, lgb_model, mlp_model = load_models()

# ── Preprocessing ─────────────────────────────────────────────
def preprocess(input_dict):
    df = pd.DataFrame([input_dict])
    for col in cat_cols:
        le  = label_encoders[col]
        val = str(df[col].values[0])
        df[col] = le.transform([val])[0] if val in le.classes_ else 0
    df[num_cols] = scaler.transform(df[num_cols])
    return df

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Online Shopper Predictor",
    page_icon="🛒",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────
st.title("🛒 Online Shopper Purchase Predictor")
st.markdown("""
This app predicts whether an online shopper will make a purchase
based on their browsing behavior.

Built with **PyTorch MLP** and **LightGBM** models trained on the
[Online Shoppers Purchasing Intention Dataset](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset).
""")

st.divider()

# ── Input Form ────────────────────────────────────────────────
st.subheader("📋 Enter Visitor Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🌐 Browsing Behavior**")
    Administrative         = st.number_input("Administrative Pages Visited", 0, 30, 0)
    Administrative_Duration = st.number_input("Time on Admin Pages (sec)", 0.0, 4000.0, 0.0)
    Informational          = st.number_input("Informational Pages Visited", 0, 25, 0)
    Informational_Duration = st.number_input("Time on Info Pages (sec)", 0.0, 3000.0, 0.0)
    ProductRelated         = st.number_input("Product Pages Visited", 0, 800, 10)
    ProductRelated_Duration = st.number_input("Time on Product Pages (sec)", 0.0, 65000.0, 500.0)

with col2:
    st.markdown("**📊 Session Metrics**")
    BounceRates  = st.slider("Bounce Rate", 0.0, 0.20, 0.02)
    ExitRates    = st.slider("Exit Rate", 0.0, 0.20, 0.04)
    PageValues   = st.number_input("Page Value Score", 0.0, 400.0, 10.0)
    SpecialDay   = st.slider("Special Day (0=normal, 1=holiday)", 0.0, 1.0, 0.0)

st.markdown("**👤 Visitor Details**")
col3, col4 = st.columns(2)

with col3:
    Month       = st.selectbox("Month", ['Feb','Mar','May','June','Jul',
                                          'Aug','Sep','Oct','Nov','Dec'])
    VisitorType = st.selectbox("Visitor Type", ['Returning_Visitor',
                                                  'New_Visitor', 'Other'])
    Weekend     = st.checkbox("Weekend Visit")

with col4:
    OperatingSystems = st.selectbox("Operating System", [1,2,3,4,5,6,7,8])
    Browser          = st.selectbox("Browser", [1,2,3,4,5,6,7,8,9,10,11,12,13])
    Region           = st.selectbox("Region", [1,2,3,4,5,6,7,8,9])
    TrafficType      = st.selectbox("Traffic Type", list(range(1,21)))

st.divider()

# ── Predict Button ────────────────────────────────────────────
if st.button("🔮 Predict Purchase Intent", use_container_width=True):

    input_dict = {
        'Administrative':           Administrative,
        'Administrative_Duration':  Administrative_Duration,
        'Informational':            Informational,
        'Informational_Duration':   Informational_Duration,
        'ProductRelated':           ProductRelated,
        'ProductRelated_Duration':  ProductRelated_Duration,
        'BounceRates':              BounceRates,
        'ExitRates':                ExitRates,
        'PageValues':               PageValues,
        'SpecialDay':               SpecialDay,
        'Month':                    Month,
        'OperatingSystems':         OperatingSystems,
        'Browser':                  Browser,
        'Region':                   Region,
        'TrafficType':              TrafficType,
        'VisitorType':              VisitorType,
        'Weekend':                  str(Weekend)
    }

    df = preprocess(input_dict)

    # MLP prediction
    cat_tensor = torch.tensor(df[cat_cols].values, dtype=torch.long)
    num_tensor = torch.tensor(df[num_cols].values, dtype=torch.float32)
    with torch.no_grad():
        logit    = mlp_model(cat_tensor, num_tensor)
        mlp_prob = torch.sigmoid(logit).item()

    # LightGBM prediction
    lgb_prob = lgb_model.predict_proba(
        df[num_cols + cat_cols])[:, 1][0]

    # ── Results ───────────────────────────────────────────────
    st.subheader("🎯 Prediction Results")

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("### 🧠 Neural Network (MLP)")
        if mlp_prob >= 0.5:
            st.success(f"✅ Will Purchase")
        else:
            st.error(f"❌ Won't Purchase")
        st.metric("Confidence", f"{mlp_prob*100:.1f}%")
        st.progress(float(mlp_prob))

    with col6:
        st.markdown("### 🌳 LightGBM")
        if lgb_prob >= 0.5:
            st.success(f"✅ Will Purchase")
        else:
            st.error(f"❌ Won't Purchase")
        st.metric("Confidence", f"{lgb_prob*100:.1f}%")
        st.progress(float(lgb_prob))

    st.divider()

    # ── Insight ───────────────────────────────────────────────
    avg_prob = (mlp_prob + lgb_prob) / 2
    st.subheader("💡 Business Insight")
    if avg_prob >= 0.6:
        st.info(f"""
        🟢 **High purchase intent detected!**
        Average confidence: {avg_prob*100:.1f}%

        This visitor is likely to buy. Consider showing:
        - Personalized product recommendations
        - Limited time offers
        - Free shipping promotion
        """)
    elif avg_prob >= 0.4:
        st.info(f"""
        🟡 **Medium purchase intent**
        Average confidence: {avg_prob*100:.1f}%

        This visitor might need a nudge. Consider showing:
        - Customer reviews
        - Discount codes
        - Exit intent popup
        """)
    else:
        st.info(f"""
        🔴 **Low purchase intent**
        Average confidence: {avg_prob*100:.1f}%

        This visitor is just browsing. Consider:
        - Newsletter signup
        - Wishlist feature
        - Retargeting ads
        """)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown("""
**Model Performance:**
| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| MLP (Neural Network) | 0.8110 | 0.5764 | 0.9039 |
| LightGBM ✅ | 0.8767 | 0.6498 | 0.9216 |

*Dataset: Online Shoppers Purchasing Intention (12,330 sessions)*
""")
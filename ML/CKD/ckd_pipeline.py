import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

df = pd.read_csv(r"D:\__Projects\Graduation---Project\ML\CKD\Data\Data.csv")

lite_numeric_features = ["gfr", "c3_c4", "bun", "blood_pressure", "serum_creatinine", "urine_ph", "months", "oxalate_levels"]
lite_categorical_features = ["stress_level", "family_history"]
lite_feature_cols = lite_numeric_features + lite_categorical_features

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), 
])

lite_preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer,     lite_numeric_features),
    ("cat", categorical_transformer, lite_categorical_features),
], remainder="drop")

X_lite = df[lite_feature_cols]
y_stage = df["ckd_stage"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_lite, y_stage, test_size=0.2, random_state=42, stratify=y_stage
)

lite_pipeline = Pipeline([
    ("pre", lite_preprocessor),
    ("xgb", XGBClassifier(random_state=42))
])


lite_pipeline.fit(X_tr, y_tr)
y_pred=lite_pipeline.predict(X_te)
print(classification_report(y_te, y_pred,
                             target_names=[f"Stage {i}" for i in range(6)]))

joblib.dump(lite_pipeline, r"D:\__Projects\Graduation---Project\app\models\ckd_stage_lite_pipeline.pkl")

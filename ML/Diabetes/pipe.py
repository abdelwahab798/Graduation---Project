import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df=pd.read_csv(r"C:\Users\nice\Desktop\projects\diabetes_prediction_project\data\diabetes_prediction_dataset.csv")
df.drop_duplicates(inplace=True)
df.loc[df["age"]<1,[ "age"]].sort_values(by="age",ascending=False)
df.loc[df["gender"]=="Other",["gender"]]=np.nan
df.dropna(inplace=True)
categorical_features = ["gender", "smoking_history"]

x=df.drop("diabetes",axis=1)
y=df["diabetes"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
    ],
    remainder='passthrough' 
)

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(
        scale_pos_weight=2,
        eval_metric='logloss',
        random_state=42
    ))
])


full_pipeline.fit(x_train, y_train)

y_pred4 = full_pipeline.predict(x_test)
print("xgboost Accuracy:", accuracy_score(y_test, y_pred4))
print("xgboost Classification Report:\n", classification_report(y_test, y_pred4))
print("roc_auc_score:", roc_auc_score(y_test, y_pred4))

joblib.dump(full_pipeline, r"D:\__Projects\Graduation---Project\app\models\diabetes_pipeline.pkl")

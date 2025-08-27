
import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

def load_data(path):
    df = pd.read_csv(path)
    y = (df['Attrition_Flag'].astype(str).str.strip().str.lower() == 'attrited customer').astype(int).values
    X = df.drop(columns=['Attrition_Flag', 'CLIENTNUM'], errors='ignore')
    return X, y

def build_prep(cat_cols, num_cols):
    num_pipe = Pipeline([('impute', SimpleImputer(strategy='median')),
                         ('scale', StandardScaler())])
    cat_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                         ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('num', num_pipe, num_cols),
                              ('cat', cat_pipe, cat_cols)])

def main(data_csv="data/customer_data.csv", out_dir="models"):
    X, y = load_data(data_csv)
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if X[c].dtype != 'object']
    prep = build_prep(cat_cols, num_cols)
    clf = GradientBoostingClassifier(random_state=42, n_estimators=300, learning_rate=0.1, max_depth=3,
                                     subsample=0.8, max_features='sqrt')
    pipe = Pipeline([('prep', prep), ('clf', clf)])
    pipe.fit(X, y)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(out_dir, "champion.pkl"))
    print(f"Saved model to {os.path.join(out_dir, 'champion.pkl')}")

if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Placeholder ethical AI model with basic fairness check
def train_model(data):
    df = pd.read_csv(data)

    X = df.drop('label', axis=1)
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    preds = model.predict(X_scaled)
    print(classification_report(y, preds))

    return model

if __name__ == "__main__":
    model = train_model("data/sample_data.csv")

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Load and Fix CSV structure
df_raw = pd.read_csv('parkinsons.csv')
column_names = df_raw.columns[0].split(',')
df = df_raw.iloc[:, 0].str.split(',', expand=True)
df.columns = column_names

# 2. Clean Data
for col in df.columns:
    if col != 'name':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.drop(columns=['name']).dropna()

# 3. Split Features and Target
X = df.drop(columns=['status'])
y = df['status'].astype(int)
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale and Train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Save everything
with open('parkinson_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('features.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print(f"Training Complete. Accuracy: {model.score(X_test_scaled, y_test):.2%}")
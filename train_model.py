import pandas as pd
import numpy as np
import joblib, json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load CSV
df = pd.read_csv("Student_Performance.csv")

features = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']
target = 'Performance Index'

# Convert to numeric and clean NaNs
for c in features + [target]:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=[target])
for c in features:
    df[c] = df[c].fillna(df[c].median())

# Convert target into categories
y_num = df[target]
try:
    df['Level'] = pd.qcut(y_num, 3, labels=['Low','Medium','High'])
except Exception:
    bins = np.quantile(y_num, [0,0.33,0.66,1.0])
    df['Level'] = pd.cut(y_num, bins=bins, labels=['Low','Medium','High'], include_lowest=True)

X = df[features]
y = df['Level']

# Split & train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_s, y_train)
print("✅ Model trained successfully.")

# Save model and scaler
joblib.dump(model, "model.pkl", compress=3)
joblib.dump(scaler, "scaler.pkl", compress=3)

# Save learning paths
learning_paths = {
    "Low": {
        "title": "Foundation Builder",
        "modules": ["Core Concepts Refresher", "Practice Tests", "Study Habit Improvement"],
        "hours": "10-15 hrs/week"
    },
    "Medium": {
        "title": "Targeted Growth",
        "modules": ["Advanced Practice", "Timed Papers", "Weekly Revision"],
        "hours": "6-10 hrs/week"
    },
    "High": {
        "title": "Advanced Mastery",
        "modules": ["Projects", "Peer Challenges", "Mock Tests"],
        "hours": "4-8 hrs/week"
    }
}
with open("learning_paths.json", "w") as f:
    json.dump(learning_paths, f, indent=2)

print("✅ Files saved: model.pkl, scaler.pkl, learning_paths.json")


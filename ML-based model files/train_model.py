# train_model.py - Run on your computer
from everywhereml.sklearn.ensemble import RandomForestClassifier

import pandas as pd
from sklearn.model_selection import train_test_split

# Load collected data
data = pd.read_csv("training_data.csv")

# Prepare features and labels
X = data[['temperature', 'humidity', 'pressure']].values
y = (data['status'] == 'Comfortable').astype(int).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
clf = RandomForestClassifier(n_estimators=10, max_depth=5)
clf.fit(X_train, y_train)

# Test accuracy
accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.1f}%")

# export to micropython
print("\nGenerating MicroPython Code\n")

# Try the preferred export method directly to micropython and save result
preferred = 'to_micropython'

if hasattr(clf, preferred):
    try:
        export_code = getattr(clf, preferred)()
        print(f"Exported using: {preferred}")
        with open("comfort_model.py", "w") as f:
            f.write(export_code)
    except Exception as e:
        print(f"Failed to export using {preferred}: {e}")

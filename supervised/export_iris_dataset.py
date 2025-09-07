from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()

# Create DataFrame with features and target
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

# Save to CSV
df.to_csv('iris_dataset.csv', index=False)

print("✅ Iris dataset exported to iris_dataset.csv")
print(f"📊 Dataset shape: {df.shape}")
print(f"🌸 Species: {list(iris.target_names)}")

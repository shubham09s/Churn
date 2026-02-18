import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

ROOT = os.path.dirname(__file__)
csv_path = os.path.join(ROOT, '..', 'customer_churn_dataset_100k.csv')

df = pd.read_csv(csv_path)

# simple churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.tight_layout()
out1 = os.path.join(ROOT, 'static', 'images', 'churn_distribution.png')
plt.savefig(out1)
plt.close()

# correlation heatmap for numeric columns
num = df.select_dtypes(include=['int64', 'float64'])
corr = num.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap (numeric features)')
plt.tight_layout()
out2 = os.path.join(ROOT, 'static', 'images', 'corr_heatmap.png')
plt.savefig(out2)
plt.close()

print('Saved:', out1)
print('Saved:', out2)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from google.colab import files
uploaded=files.upload()
import statistics
import io
data=pd.read_csv(io.BytesIO(uploaded['Disease_symptom_and_patient_profile_dataset.csv']))
data
print(data.head())
print(data.tail())
print(data.info())
print(data.describe())
print(data.isnull().sum())
data = data.dropna()
print(data['Fever'].unique())
transposed_data = data.T
transposed_data
data['Disease_Fever'] = data['Disease'].str.cat(data['Fever'], sep=' - ')
data
import seaborn as sns
import matplotlib.pyplot as plt
#univariate analysis
sns.histplot(data['Fever'],bins=20)
plt.show()
#bivariate analysis
sns.scatterplot(x='Fever', y='Cough', data=data)
plt.show()
#multivariate analysis
sns.pairplot(data)
plt.show()
grouped_data = data.groupby('Disease')
aggregated_data = grouped_data.agg({'Age': 'mean'})
aggregated_data
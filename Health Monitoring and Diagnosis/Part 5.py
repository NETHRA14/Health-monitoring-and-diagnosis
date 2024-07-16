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
import matplotlib.pyplot as plt 
plt.hist(data['Age'], bins=20)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Numerical Column')
plt.show()
plt.bar(data['Age'].value_counts().index,
data['Age'].value_counts().values)
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.title('Bar Chart of Category Column')
plt.show()
plt.scatter(data['Age'], data['Blood Pressure'])
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.title('Scatter Plot of Age vs Blood Pressure')
plt.show()
import seaborn as sns 
sns.boxplot(x='Age', y='Blood Pressure', data=data)
plt.xlabel('Category')
plt.ylabel('Numerical Column')
plt.title('Box Plot of Numerical Column by Category')
plt.show()
sns.pairplot(data)
plt.title('Pair Plot of Numerical Variables')
plt.show()
import plotly.express as px 
fig = px.scatter(data, x='Age', y='Blood Pressure',
hover_data=['Gender'])
fig.show()
import dash 
import dash_core_components as dcc 
import dash_html_components as html 
app = dash.Dash(_name_)
app.layout = html.Div([
dcc.Graph(
id='interactive-plot',
figure={
'data': [
{'x': data['Age'], 'y': data['Blood Pressure'],
'mode': 'markers', 'type': 'scatter'}
],
'layout': {
'title': 'Interactive Scatter Plot',
'xaxis': {'title': 'Age'},
'yaxis': {'title': 'Blood Pressure'}
}
}
)
])
if _name_ == '_main_':
app.run_server(debug=True)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
# Load data
data = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')
# Handle missing values
data.fillna(method='ffill', inplace=True)
# Encoding categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
 le = LabelEncoder()
 data[column] = le.fit_transform(data[column])
 label_encoders[column] = le
# Split data into features and target
X = data.drop('Age', axis=1)
y = data['Age']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Exploratory Data Analysis
sns.histplot(y, kde=True)
plt.title('Price Distribution')
plt.show()
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
# Initialize models
models = {
 'Linear Regression': LinearRegression(),
 'Random Forest': RandomForestRegressor(random_state=42),
 'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}
# Train models
for name, model in models.items():
 model.fit(X_train, y_train)
 print(f"{name} trained.")
# Evaluate models
results = {}
for name, model in models.items():
 y_pred = model.predict(X_test)
 results[name] = {
 'RMSE': mean_squared_error(y_test, y_pred, squared=False),
 'MAE': mean_absolute_error(y_test, y_pred),
 'R^2': r2_score(y_test, y_pred)
 }
# Print evaluation results
for name, metrics in results.items():
 print(f"Model: {name}")
 for metric, value in metrics.items():
 print(f"{metric}: {value}")
 print("\n")
# Hyperparameter tuning for Random Forest
param_grid = {
 'n_estimators': [100, 200, 300],
 'max_depth': [None, 10, 20, 30]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42),
param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(f"Best parameters for Random Forest:{grid_search.best_params_}")
# Evaluate the tuned model
y_pred = best_rf.predict(X_test)
tuned_results = {
 'RMSE': mean_squared_error(y_test, y_pred, squared=False),
 'MAE': mean_absolute_error(y_test, y_pred),
 'R^2': r2_score(y_test, y_pred)
}
print("Tuned Random Forest performance:")
for metric, value in tuned_results.items():
 print(f"{metric}: {value}")
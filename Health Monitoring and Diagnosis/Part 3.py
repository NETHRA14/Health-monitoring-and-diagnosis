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
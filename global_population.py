import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Loading the dataset
data = pd.read_csv('../input/world-population-dataset/world_population.csv')

# Checking the first rows of the DataFrame
print(data.head())

# Getting the number of rows and columns
num_rows = len(data)
num_cols = len(data.columns)

print(num_rows)
print(num_cols)

# Summary statistics
summary_stats = data.describe().transpose()
summary_stats = summary_stats.sort_values(by='mean', ascending=False)

# Visualization using seaborn
sns.set_palette("BuGn")
sns.barplot(x=summary_stats['mean'], y=summary_stats.index, hue=summary_stats.index.isin(['std']))
plt.title("Descriptive Statistics Ordered by Mean")
plt.show()

# Number of unique values in each column
for feature in data.columns:
    unique_values = data[feature].nunique()
    print(f"{feature} ---> {unique_values}")

# Missing values
missing_values = data.isna().sum()
print(missing_values)

# Missing data analysis
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cmap=['blue', 'red'])
plt.title("Missing Data Analysis")
plt.show()

# Statistics by continent
continent_data = data.groupby('Continent').mean().sort_values(by='Density (per km²)', ascending=False)

# Visualization using seaborn
sns.heatmap(continent_data, cmap="BuGn")
plt.title("Continent-wise Statistics")
plt.show()

# Visualization using plotly
fig = px.line(continent_data.T, x=continent_data.columns, y=continent_data.index, color_discrete_sequence=px.colors.qualitative.Set1)
fig.update_layout(title="Population Over Time by Continent", xaxis_title="Continent", yaxis_title="Population", legend_title="Year")
fig.show()

# Interactive visualization using plotly.graph_objects
fig = go.Figure(data=go.Scattergeo(
    locations=data['Country/Territory'],
    text=data['Country/Territory'],
    marker=dict(
        size=data['2022 Population'],
        colorscale='Viridis',
        colorbar_title="2022 Population"
    )
))

fig.update_layout(title="2022 Population", geo=dict(showframe=False, projection_type='mercator'))
fig.show()

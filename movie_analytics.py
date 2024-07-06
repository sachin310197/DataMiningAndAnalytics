import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt

# Load Data
movies_df = pd.read_csv('./Movies.csv', encoding='ISO-8859-1')
ratings_df = pd.read_csv('./Ratings.csv', encoding='ISO-8859-1')
users_df = pd.read_csv('./Users.csv', encoding='ISO-8859-1')

# Clean Data
# Extract year from movie title and split categories
movies_df['Year'] = movies_df['Title'].str.extract(r'\((\d{4})\)')
movies_df['Year'] = movies_df['Year'].astype(float).fillna(0).astype(int, errors='ignore')
movies_df['Category'] = movies_df['Category'].str.split('|')

# Filter ratings
ratings_df = ratings_df[(ratings_df['Rating'] >= 1) & (ratings_df['Rating'] <= 5)]

# Merge DataFrames
merged_df = pd.merge(ratings_df, movies_df, on='MovieID')
merged_df = merged_df.explode('Category')
merged_df = pd.merge(merged_df, users_df, on='UserID')

# Analysis and Queries
# 1. Total number of movies released each year
movies_per_year = movies_df['Year'].value_counts().sort_index()

# 2. Movie category with highest ratings each year
grouped_df = merged_df.groupby(['Year', 'Category'])['Rating'].mean().reset_index()
highest_rated_category_per_year = grouped_df.loc[grouped_df.groupby('Year')['Rating'].idxmax()]

# 3. Movie category and age group wise likings
age_group_likings = merged_df.groupby(['Age', 'Category'])['Rating'].count().reset_index()
most_liked_categories_per_age_group = age_group_likings.loc[age_group_likings.groupby('Age')['Rating'].idxmax()]

# 4. Clustering: Movie category and age group wise likings
pivot_table = merged_df.pivot_table(index='UserID', columns='Category', values='Rating', fill_value=0)
print("Pivot table shape:", pivot_table.shape)

kmeans = KMeans(n_clusters=5, random_state=0).fit(pivot_table)
print("KMeans labels length:", len(kmeans.labels_))

if len(kmeans.labels_) == merged_df['UserID'].nunique():
    user_id_to_cluster = pd.Series(kmeans.labels_, index=pivot_table.index)
    merged_df['Cluster'] = merged_df['UserID'].map(user_id_to_cluster)
else:
    raise ValueError("Mismatch in the number of clusters and unique UserIDs")

# Visualize clusters
fig, ax = plt.subplots()
ax.scatter(merged_df['Age'], merged_df['Cluster'])
ax.set_xlabel('Age')
ax.set_ylabel('Cluster')
ax.set_title('Age vs Cluster')
st.pyplot(fig)

# 5. Year wise count of movies released
yearly_movie_count = movies_df['Year'].value_counts().sort_index()

# 6. Year and category wise count of movies released
year_category_count = merged_df.groupby(['Year', 'Category']).size().reset_index(name='Count')

# 7. Clustering: Movie category and occupation
pivot_table_occupation = merged_df.pivot_table(index='Occupation', columns='Category', values='Rating', fill_value=0)
kmeans_occupation = KMeans(n_clusters=5, random_state=0).fit(pivot_table_occupation)

# Map occupation clusters to merged_df
occupation_cluster_map = pd.Series(kmeans_occupation.labels_, index=pivot_table_occupation.index)
merged_df['OccupationCluster'] = merged_df['Occupation'].map(occupation_cluster_map)

# 8. Refine predictive model: Include age group
pivot_table_age_occupation = merged_df.pivot_table(index=['Age', 'Occupation'], columns='Category', values='Rating', fill_value=0)
kmeans_age_occupation = KMeans(n_clusters=5, random_state=0).fit(pivot_table_age_occupation)

# Map age and occupation clusters to merged_df
age_occupation_cluster_map = pd.Series(kmeans_age_occupation.labels_, index=pivot_table_age_occupation.index)
merged_df['AgeOccupationCluster'] = merged_df.set_index(['Age', 'Occupation']).index.map(age_occupation_cluster_map)

# 9. Predictive model: Category to age group and occupation
# Prepare data
X = pd.get_dummies(merged_df[['Category']], columns=['Category'])
y = merged_df[['Age', 'Occupation']].applymap(str)
X = X.join(pd.get_dummies(y['Age'], prefix='Age'))
X = X.join(pd.get_dummies(y['Occupation'], prefix='Occupation'))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = MultiOutputClassifier(RandomForestClassifier())
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Model performance:")
print(model.score(X_test, y_test))

# Simple User Interface with Streamlit
st.title("Movie Analytics Dashboard")

st.sidebar.header("Queries")
query = st.sidebar.selectbox("Choose a query", [
    "Total number of movies released each year",
    "Movie category with highest ratings each year",
    "Movie category and age group wise likings",
    "Clustering: Movie category and age group wise likings",
    "Year wise count of movies released",
    "Year and category wise count of movies released",
    "Clustering: Movie category and occupation",
    "Refined predictive model: Include age group",
    "Predictive model: Category to age group and occupation"
])

if query == "Total number of movies released each year":
    st.write(movies_per_year)

elif query == "Movie category with highest ratings each year":
    st.write(highest_rated_category_per_year)

elif query == "Movie category and age group wise likings":
    st.write(most_liked_categories_per_age_group)

elif query == "Clustering: Movie category and age group wise likings":
    st.write(merged_df[['Age', 'Cluster']].drop_duplicates())

elif query == "Year wise count of movies released":
    st.write(yearly_movie_count)

elif query == "Year and category wise count of movies released":
    st.write(year_category_count)

elif query == "Clustering: Movie category and occupation":
    st.write(merged_df[['Occupation', 'OccupationCluster']].drop_duplicates())

elif query == "Refined predictive model: Include age group":
    st.write(merged_df[['Age', 'Occupation', 'AgeOccupationCluster']].drop_duplicates())

elif query == "Predictive model: Category to age group and occupation":
    st.write("Model score: ", model.score(X_test, y_test))
    category = st.text_input("Enter movie category:")
    if category:
        category_encoded = pd.get_dummies(pd.DataFrame([[category]], columns=['Category']), columns=['Category'])
        category_encoded = category_encoded.reindex(columns=X.columns, fill_value=0)
        prediction = model.predict(category_encoded)
        st.write("Predicted age group and occupation:", prediction)

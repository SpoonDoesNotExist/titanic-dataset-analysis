import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Titanic dataset", page_icon="🚢")

st.image("data/image.png")

st.title("🧊💑🚢 Titanic dataset")
st.write(
    """
    Вариант 19.
    Подсчитать количество погибших мужчин *старше указанного возраста* по каждому пункту
посадки.
    """
)


# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_data
def load_data():
    df = pd.read_csv("data/titanic_train.csv")
    return df


df = load_data()

# Show a multiselect widget with the genres using `st.multiselect`.
# genres = st.multiselect(
#     "Genres",
#     df.genre.unique(),
#     ["Action", "Adventure", "Biography", "Comedy", "Drama", "Horror"],
# )

# Show a slider widget with the years using `st.slider`.
# years = st.slider("Years", 1986, 2006, (2000, 2016))
age_limit = st.slider("Select the minimum age:", min_value=0, max_value=int(df['Age'].max())+10, value=25)

all_embarked = df['Embarked'].dropna().unique().tolist()
df['NotSurvived'] = 1 - df['Survived']

df_reshaped = df[
    (df['Sex']=='male') & (df['Age']>age_limit)
].groupby('Embarked')['NotSurvived']\
    .sum().reindex(all_embarked, fill_value=0).sort_values(ascending=False)


st.dataframe(
    df_reshaped,
    use_container_width=True,
    column_config={"Embarked": st.column_config.TextColumn("Embarked")},
)


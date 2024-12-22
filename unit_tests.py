import pytest
import pandas as pd
from io import StringIO


data = """PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
"""

@pytest.fixture
def titanic_df():
    """Фикстура для создания тестового DataFrame."""
    return pd.read_csv(StringIO(data))

def test_load_data(mocker):
    """Тест функции load_data."""
    mocker.patch("pandas.read_csv", return_value=pd.read_csv(StringIO(data)))
    from main import load_data
    df = load_data()
    assert not df.empty, "Данные не загружаются"
    assert len(df) == 6, "Некорректное количество строк"

def test_data_processing(titanic_df):
    """Тест обработки данных."""
    age_limit = 25
    titanic_df['NotSurvived'] = 1 - titanic_df['Survived']

    all_embarked = titanic_df['Embarked'].dropna().unique().tolist()
    df_reshaped = titanic_df[
        (titanic_df['Sex'] == 'male') & (titanic_df['Age'] > age_limit)
    ].groupby('Embarked')['NotSurvived']\
        .sum().reindex(all_embarked, fill_value=0).sort_values(ascending=False)

    assert df_reshaped.loc['S'] == 1, "Некорректное значение для Embarked 'S'"
    assert df_reshaped.loc['Q'] == 0, "Некорректное значение для Embarked 'Q'"
    assert len(df_reshaped) == 3, "Некорректное количество пунктов посадки"

if __name__ == "__main__":
    pytest.main()

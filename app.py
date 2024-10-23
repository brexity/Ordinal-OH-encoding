import pandas as pd
import numpy as np


def ordinal_encode(df, column, categories):
    category_to_number = {}
    for i in range(len(categories)):
        category_to_number[categories[i]] = i

    encoded_column = []
    for item in df[column]:
        encoded_column.append(category_to_number[item])

    return pd.Series(encoded_column)


def one_hot_encode(df, column):
    unique_values = df[column].unique()
    one_hot_encoded_df = pd.DataFrame()

    for value in unique_values:
        one_hot_encoded_df[column + '_' + value] = np.where(df[column] == value, 1, 0)

    df = df.drop(column, axis=1)
    df = pd.concat([df, one_hot_encoded_df], axis=1)

    return df


data = {'color': ['red', 'blue', 'green', 'blue', 'red', 'green', 'green']}
df = pd.DataFrame(data)

categories = ['red', 'blue', 'green']
df['color_ordinal'] = ordinal_encode(df, 'color', categories)
print(df)

df_one_hot = one_hot_encode(df, 'color')
print(df_one_hot)

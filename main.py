# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import requests
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# %%
# Retrieve data from API
with open("config/api.json", "r") as api_file:
    api_key = json.loads(api_file.read())["apiKey"]

headers = {"Authorization": f"ApeKey {api_key}"}
response = requests.get("https://api.monkeytype.com/results", headers=headers)

if response.status_code == 200:
    data = response.json()["data"]

# %%
df = pd.DataFrame(data)

df = df[(df['mode'] == 'time') & (df['testDuration'].isin([30, 60]))]

df['isPb'] = df['isPb'].apply(lambda x: False if x is np.nan else True)

char_columns = [
    'chars_correct',
    'chars_incorrect',
    'chars_extra',
    'chars_missed'
]
df[char_columns] = pd.DataFrame(df['charStats'].to_list())

df['timestamp'] = (
    pd.to_datetime(
        df['timestamp'],
        utc=True,
        unit='ms'
    ).dt.tz_convert(tz='America/Sao_Paulo')
)

df['date'] = df['timestamp'].dt.date

df = df[df['date'] >= date(2025, 5, 30)]

inf_quantile = df['wpm'].quantile(0.01)
upp_quantile = df['wpm'].quantile(0.99)

df = df[(df['wpm'] >= inf_quantile) & (df['wpm'] <= upp_quantile)]

df.head()

# %%
last_training_date = df['date'].max()

comb_dfs = pd.concat([
    df[df['date'] < last_training_date][['wpm']].assign(group='Before'),
    df[['wpm']].assign(group='After')
])
df_before = comb_dfs[comb_dfs['group'] == 'Before']

# %%
print("WPM before and after the last training session:")
pd.concat(
    [
        df_before['wpm'].rename('wpm_before').describe(),
        df['wpm'].rename('wpm_after').describe()
    ], axis=1
)

# %%
plt.title("Comparison of statistics before and after the last training session")
sns.boxplot(data=comb_dfs, y='wpm', x='group')
plt.xlabel("Moment of training")
plt.ylabel("WPM")
plt.grid()
plt.show()

# %%
daily_mean = (
    df.groupby('date', as_index=False).agg({
        'wpm': 'mean',
        'rawWpm': 'mean',
        '_id': 'count'
    })
)
daily_mean.columns = ['date', 'wpm', 'rawWpm', 'count']

plt.scatter(df['date'], df['wpm'])
plt.plot(daily_mean['date'], daily_mean['wpm'], color='red')
plt.grid()
plt.title('Daily WPM average')
plt.xlabel('Date')
plt.ylabel('WPM')
plt.xticks(rotation=45, ha='right')
plt.legend(['Observed', 'Average'])
plt.show()

# %%
amount_mean = daily_mean.groupby('count', as_index=False)['wpm'].mean().sort_values('count')

plt.plot(amount_mean['count'], amount_mean['wpm'])
plt.grid()
plt.title('Average WPM by amount of testes taken')
plt.xlabel('Tests taken')
plt.ylabel('WPM')
plt.show()

# %%
plt.plot(daily_mean['date'], daily_mean['wpm'])
plt.plot(daily_mean['date'], daily_mean['rawWpm'])
plt.grid()
plt.title('Impact of mistakes comparing average WPM vs. Raw WPM')
plt.xlabel('Date')
plt.ylabel('WPM')
plt.legend(['WPM', 'Raw WPM'])
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
wpm_consis = df[['wpm', 'consistency']].sort_values(['consistency', 'wpm'])

# %%
y = df['wpm']
X = df[['consistency']]

model_reg = LinearRegression()
model_tree = DecisionTreeRegressor(max_depth=3)
model_reg.fit(X, y)
model_tree.fit(X, y)

# %%
predict_reg = model_reg.predict(X.drop_duplicates())
predict_tree = model_tree.predict(X.drop_duplicates().sort_values('consistency'))

plt.plot(wpm_consis['consistency'], wpm_consis['wpm'], color='lightblue')
plt.plot(X.drop_duplicates()['consistency'], predict_reg, color='red')
plt.plot(X.drop_duplicates()['consistency'].sort_values(), predict_tree, color='green')
plt.grid()
plt.suptitle('Consistency by average WPM')
plt.title('With regressions in comparison for tendency')
plt.xlabel('Consistency')
plt.ylabel('WPM')
plt.legend([
    'Observed',
    'Linear regression',
    'Decision Tree Regression'
])
plt.show()

# %%
print("Correlation of consistency and WPM")
wpm_consis.corr()

# %%

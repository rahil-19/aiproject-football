import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

files = ["data/E0_2023.csv", "data/E0_2024.csv", "data/E0_2025.csv"]

df_list = [pd.read_csv(file) for file in files]
df = pd.concat(df_list, ignore_index=True)

df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
home_avg = df.groupby('HomeTeam')['FTHG'].mean()
away_avg = df.groupby('AwayTeam')['FTAG'].mean()

df['HomeAttack'] = df['HomeTeam'].map(home_avg)
df['AwayAttack'] = df['AwayTeam'].map(away_avg)
X = df[['HomeAttack', 'AwayAttack']]
y = df['FTR']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

def predict_match(home_team, away_team):
    home_attack = home_avg[home_team]
    away_attack = away_avg[away_team]

    X_new = pd.DataFrame([[home_attack, away_attack]], 
                         columns=['HomeAttack', 'AwayAttack'])

    pred = model.predict(X_new)
    prob = model.predict_proba(X_new)
    res = ''
    if (pred[0] == 'A'): res = away_team
    elif (pred[0] == 'H'): res = home_team
    else: res = 'Draw'

    return res, prob

joblib.dump((model, home_avg.to_dict(), away_avg.to_dict()), "model.pkl")

""" result, prob = predict_match("Man United", "Man City")

print("Prediction:", result)
print("Probabilities:", prob) """

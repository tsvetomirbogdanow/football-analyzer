import glob
import pandas as pd
import numpy as np

# Зареждане на CSV файлове
csv_files = glob.glob("data/*.csv")
if len(csv_files) == 0:
    csv_files = glob.glob("../data/*.csv")

dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file, dayfirst=True, encoding='utf-8')
        required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        if all(col in df.columns for col in required_cols):
            dfs.append(df)
    except:
        pass

if len(dfs) == 0:
    raise Exception("❌ Няма валидни CSV файлове в data/ !")

data = pd.concat(dfs, ignore_index=True)

# Отбори
teams = sorted(set(data['HomeTeam']).union(data['AwayTeam']))

# Средни голове
mean_home_goals = data['FTHG'].mean()
mean_away_goals = data['FTAG'].mean()

def weighted_avg(df, col, last_matches=10):
    df_tail = df.tail(last_matches)
    if len(df_tail) == 0 or col not in df_tail.columns:
        return np.nan
    weights = np.exp(np.linspace(-1, 0, len(df_tail)))
    return np.average(df_tail[col].fillna(0), weights=weights)

def team_strength(team, is_home, last_matches=10):
    df_team = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)]
    if df_team.empty:
        return 1.0, 1.0, np.nan

    if is_home:
        goals_for = weighted_avg(df_team[df_team['HomeTeam'] == team], 'FTHG', last_matches)
        goals_against = weighted_avg(df_team[df_team['AwayTeam'] == team], 'FTAG', last_matches)
    else:
        goals_for = weighted_avg(df_team[df_team['AwayTeam'] == team], 'FTAG', last_matches)
        goals_against = weighted_avg(df_team[df_team['HomeTeam'] == team], 'FTHG', last_matches)

    league_off = mean_home_goals if is_home else mean_away_goals
    attack = (goals_for / league_off) if league_off and league_off>0 else 1.0
    defense = (goals_against / (mean_away_goals if is_home else mean_home_goals)) if (mean_away_goals if is_home else mean_home_goals) else 1.0

    return float(attack), float(defense), np.nan

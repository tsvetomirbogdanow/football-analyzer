from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from scipy.stats import poisson
import glob
import os

app = Flask(__name__)

# === Зареждане на CSV файлове ===
csv_files = glob.glob("data/*.csv")
dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        if all(col in df.columns for col in required_cols):
            dfs.append(df)
            print(f"✅ Зареден файл: {file} ({len(df)} реда)")
        else:
            print(f"⚠️ Пропуснат файл (липсващи колони): {file}")
    except Exception as e:
        print(f"❌ Грешка при зареждане на {file}: {e}")

if len(dfs) == 0:
    raise Exception("❌ Няма валидни CSV файлове!")

data = pd.concat(dfs, ignore_index=True)
teams = sorted(set(data['HomeTeam']).union(data['AwayTeam']))

# --- Средни за лигата ---
mean_home_goals = data['FTHG'].mean()
mean_away_goals = data['FTAG'].mean()

# --- Функция за тежено средно (форма) ---
def weighted_avg(df, col, last_matches=10):
    df_tail = df.tail(last_matches)
    if len(df_tail) == 0:
        return 1.0
    weights = np.exp(np.linspace(-1,0,len(df_tail)))
    return np.average(df_tail[col], weights=weights)

# --- Attack/Defense ratio ---
def team_strength(team, is_home, last_matches=10):
    if is_home:
        df_team = data[(data['HomeTeam']==team) | (data['AwayTeam']==team)]
        attack = weighted_avg(df_team[df_team['HomeTeam']==team], 'FTHG', last_matches)
        defense = weighted_avg(df_team[df_team['AwayTeam']==team], 'FTAG', last_matches)
    else:
        df_team = data[(data['HomeTeam']==team) | (data['AwayTeam']==team)]
        attack = weighted_avg(df_team[df_team['AwayTeam']==team], 'FTAG', last_matches)
        defense = weighted_avg(df_team[df_team['HomeTeam']==team], 'FTHG', last_matches)
    return attack, defense

# --- H2H корекция ---
def h2h_correction(home, away, max_adjust=0.2):
    df_h2h = data[((data["HomeTeam"]==home)&(data["AwayTeam"]==away)) |
                  ((data["HomeTeam"]==away)&(data["AwayTeam"]==home))]
    if len(df_h2h)==0:
        return 1.0,1.0
    df_h2h = df_h2h.tail(5)
    home_scores = df_h2h[(df_h2h['HomeTeam']==home)]['FTHG'].mean() - df_h2h[(df_h2h['HomeTeam']==home)]['FTAG'].mean()
    away_scores = df_h2h[(df_h2h['HomeTeam']==away)]['FTHG'].mean() - df_h2h[(df_h2h['HomeTeam']==away)]['FTAG'].mean()
    home_corr = 1 + np.clip(home_scores*0.05, -max_adjust, max_adjust)
    away_corr = 1 + np.clip(away_scores*0.05, -max_adjust, max_adjust)
    return home_corr, away_corr

# --- Основна прогноза ---
def predict_match(home, away, simulations=3000):
    if home == away:
        return "⚠️ Моля, избери два различни отбора."

    # --- Attack/Defense ---
    home_attack, home_defense = team_strength(home, True)
    away_attack, away_defense = team_strength(away, False)

    # Проверка за NaN или нули
    home_attack = 1 if (home_attack is None or np.isnan(home_attack) or home_attack <= 0) else home_attack
    home_defense = 1 if (home_defense is None or np.isnan(home_defense) or home_defense <= 0) else home_defense
    away_attack = 1 if (away_attack is None or np.isnan(away_attack) or away_attack <= 0) else away_attack
    away_defense = 1 if (away_defense is None or np.isnan(away_defense) or away_defense <= 0) else away_defense

    # --- H2H ---
    home_corr, away_corr = h2h_correction(home, away)
    home_corr = 1 if np.isnan(home_corr) else home_corr
    away_corr = 1 if np.isnan(away_corr) else away_corr

    # --- Лямбда за Poisson ---
    lambda_home = mean_home_goals * (home_attack / away_defense) * home_corr
    lambda_away = mean_away_goals * (away_attack / home_defense) * away_corr
    lambda_home = np.clip(lambda_home, 0.2, 4.5)
    lambda_away = np.clip(lambda_away, 0.2, 4.5)

    # --- Симулации за победа/равенство/загуба и BTTS ---
    home_wins = draws = away_wins = btts = 0
    score_counter = {}

    for _ in range(simulations):
        gh = np.random.poisson(lambda_home)
        ga = np.random.poisson(lambda_away)

        if gh > ga:
            home_wins += 1
        elif gh == ga:
            draws += 1
        else:
            away_wins += 1

        if gh > 0 and ga > 0:
            btts += 1

        score_counter[(gh, ga)] = score_counter.get((gh, ga), 0) + 1

    home_win_prob = home_wins / simulations * 100
    draw_prob = draws / simulations * 100
    away_win_prob = away_wins / simulations * 100
    btts_prob = btts / simulations * 100
    most_likely_score = max(score_counter, key=score_counter.get)

    # --- Over/Under общи голове ---
    lambda_total = lambda_home + lambda_away
    over_1_5 = 1 - poisson.cdf(1, lambda_total)
    over_2_5 = 1 - poisson.cdf(2, lambda_total)
    over_3_5 = 1 - poisson.cdf(3, lambda_total)

    return f"""
    <div class='result-box'>
      <h2>{home} 🆚 {away}</h2>
      <p><b>Най-вероятен резултат:</b> {most_likely_score[0]} - {most_likely_score[1]}</p>
      <div class='probabilities'>
        <p>🏠 Победа за {home}: <b>{home_win_prob:.1f}%</b></p>
        <p>🤝 Равенство: <b>{draw_prob:.1f}%</b></p>
        <p>🚀 Победа за {away}: <b>{away_win_prob:.1f}%</b></p>
        <p>⚡ BTTS (И двата отбора отбелязват): <b>{btts_prob:.1f}%</b></p>
        <hr>
        <p>Over/Under (Общо голове в мача):</p>
        <p>Over 1.5: <b>{over_1_5*100:.1f}%</b></p>
        <p>Over 2.5: <b>{over_2_5*100:.1f}%</b></p>
        <p>Over 3.5: <b>{over_3_5*100:.1f}%</b></p>
      </div>
    </div>
    """

@app.route("/", methods=["GET","POST"])
def index():
    result = None
    selected_home = None
    selected_away = None
    if request.method == "POST":
        selected_home = request.form.get("home_team")
        selected_away = request.form.get("away_team")
        result = predict_match(selected_home, selected_away)
    return render_template("index.html", teams=teams, result=result, selected_home=selected_home, selected_away=selected_away)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

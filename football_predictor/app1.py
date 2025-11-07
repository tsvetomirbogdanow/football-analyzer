from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from scipy.stats import poisson
import glob

app = Flask(__name__)

# === Зареждане на CSV файловете ===
csv_files = glob.glob("data/*.csv")
dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        if all(col in df.columns for col in ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']):
            dfs.append(df)
            print(f"✅ Зареден файл: {file} ({len(df)} реда)")
        else:
            print(f"⚠️ Пропуснат файл (липсваща колона): {file}")
    except Exception as e:
        print(f"❌ Грешка при зареждане на {file}: {e}")

if len(dfs) == 0:
    raise Exception("❌ Няма намерени валидни CSV файлове в папката!")

data = pd.concat(dfs, ignore_index=True)
teams = sorted(set(data['HomeTeam']).union(data['AwayTeam']))

# === Функция за експоненциални тежести на последните мачове ===
def weighted_stats_exp(df, team, is_home, last_matches=10):
    if is_home:
        df_team = df[df["HomeTeam"] == team][["FTHG", "FTAG"]]
    else:
        df_team = df[df["AwayTeam"] == team][["FTAG", "FTHG"]]

    if len(df_team) == 0:
        return (1, 1)

    df_team = df_team.tail(last_matches)
    weights = np.exp(np.linspace(-1, 0, len(df_team)))  # експоненциални тежести
    avg_scored = np.average(df_team.iloc[:, 0], weights=weights)
    avg_conceded = np.average(df_team.iloc[:, 1], weights=weights)
    return (avg_scored, avg_conceded)

# === Функция за история на директните срещи (H2H) ===
def h2h_adjustment(home, away, last_h2h=5):
    df_h2h = data[((data["HomeTeam"] == home) & (data["AwayTeam"] == away)) |
                  ((data["HomeTeam"] == away) & (data["AwayTeam"] == home))]
    if len(df_h2h) == 0:
        return 1.0  # няма корекция
    df_h2h = df_h2h.tail(last_h2h)
    home_adv = 1 + (df_h2h['FTHG'][df_h2h['HomeTeam'] == home].sum() - df_h2h['FTAG'][df_h2h['HomeTeam'] == home].sum())*0.05
    return max(0.8, min(home_adv, 1.2))  # ограничаваме корекцията

# === Предсказване на мач с Poisson и Monte Carlo симулация ===
def predict_match(home, away, simulations=10000):
    if home == away:
        return "⚠️ Моля, избери два различни отбора."

    last_matches = 10

    # --- Сила на отборите ---
    home_attack, home_defense = weighted_stats_exp(data, home, is_home=True, last_matches=last_matches)
    away_attack, away_defense = weighted_stats_exp(data, away, is_home=False, last_matches=last_matches)

    # --- Средни голове в лигата ---
    mean_home_goals = data['FTHG'].mean()
    mean_away_goals = data['FTAG'].mean()

    # --- Коефициенти за сила ---
    attack_strength_home = home_attack / mean_home_goals
    defense_strength_home = home_defense / mean_away_goals
    attack_strength_away = away_attack / mean_away_goals
    defense_strength_away = away_defense / mean_home_goals

    # --- Очаквани голове λ ---
    lambda_home = mean_home_goals * attack_strength_home * defense_strength_away * h2h_adjustment(home, away)
    lambda_away = mean_away_goals * attack_strength_away * defense_strength_home

    # --- Monte Carlo симулация ---
    home_wins = 0
    draws = 0
    away_wins = 0
    over_1_5 = 0
    over_2_5 = 0
    over_3_5 = 0
    most_common_score = {}

    for _ in range(simulations):
        goals_home = np.random.poisson(lambda_home)
        goals_away = np.random.poisson(lambda_away)

        # Победа/равенство
        if goals_home > goals_away:
            home_wins += 1
        elif goals_home == goals_away:
            draws += 1
        else:
            away_wins += 1

        # Over/Under
        total_goals = goals_home + goals_away
        if total_goals > 1:
            over_1_5 += 1
        if total_goals > 2:
            over_2_5 += 1
        if total_goals > 3:
            over_3_5 += 1

        # Най-често срещан резултат
        key = (goals_home, goals_away)
        most_common_score[key] = most_common_score.get(key, 0) + 1

    # --- Вероятности ---
    home_win_prob = home_wins / simulations
    draw_prob = draws / simulations
    away_win_prob = away_wins / simulations
    over_1_5_prob = over_1_5 / simulations
    over_2_5_prob = over_2_5 / simulations
    over_3_5_prob = over_3_5 / simulations

    # Най-вероятен резултат
    most_likely_score = max(most_common_score, key=most_common_score.get)

    # --- HTML резултат ---
    return f"""
    <div class='result-box'>
      <h2>{home} 🆚 {away}</h2>
      <p><b>Най-вероятен резултат:</b> {most_likely_score[0]} - {most_likely_score[1]}</p>
      <div class='probabilities'>
        <p>🏠 Победа за {home}: <b>{home_win_prob*100:.1f}%</b></p>
        <p>🤝 Равенство: <b>{draw_prob*100:.1f}%</b></p>
        <p>🚀 Победа за {away}: <b>{away_win_prob*100:.1f}%</b></p>
        <p>⚡ Over 1.5 гола: <b>{over_1_5_prob*100:.1f}%</b></p>
        <p>⚡ Over 2.5 гола: <b>{over_2_5_prob*100:.1f}%</b></p>
        <p>⚡ Over 3.5 гола: <b>{over_3_5_prob*100:.1f}%</b></p>
      </div>
    </div>
    """

# === Flask route ===
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        home_team = request.form.get("home_team")
        away_team = request.form.get("away_team")
        result = predict_match(home_team, away_team)
    return render_template("index.html", teams=teams, result=result)

if __name__ == "__main__":
    app.run(debug=True, port=8080)

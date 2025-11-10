import pandas as pd
import numpy as np
from scipy.stats import poisson

# --- Вътрешни функции ---

def calculate_implied_prob(odds):
    """
    Превръща коефициент в имплицитна вероятност.
    """
    if odds <= 0:
        return 0.0
    return 1.0 / odds

def normalize_probs(probs):
    """
    Нормализира вероятности, така че сумата им да е 1.
    """
    total = sum(probs)
    if total > 0:
        return [p/total for p in probs]
    return [0.0]*len(probs)

def find_value_bets(home, away, data, min_edge=0.05):
    """
    Връща HTML с value bets за конкретен мач.
    """
    # Унифициран филтър: премахваме интервали и правим lowercase
    df_match = data[
        (data['HomeTeam'].str.strip().str.lower() == home.strip().lower()) &
        (data['AwayTeam'].str.strip().str.lower() == away.strip().lower())
    ]

    if df_match.empty:
        return "<p>Няма налични данни за този мач.</p>"

    df_match = df_match.iloc[0]  # вземаме първия ред

    # Имплицитни вероятности от средни коефициенти
    prob_H = calculate_implied_prob(df_match.get("AvgH", 0))
    prob_D = calculate_implied_prob(df_match.get("AvgD", 0))
    prob_A = calculate_implied_prob(df_match.get("AvgA", 0))
    prob_H, prob_D, prob_A = normalize_probs([prob_H, prob_D, prob_A])

    # Реални вероятности чрез Poisson (средни голове от DataFrame)
    mean_home_goals = data['FTHG'].mean()
    mean_away_goals = data['FTAG'].mean()

    lambda_home = mean_home_goals * (df_match['FTHG'] / mean_home_goals if df_match['FTHG']>0 else 1)
    lambda_away = mean_away_goals * (df_match['FTAG'] / mean_away_goals if df_match['FTAG']>0 else 1)

    simulations = 5000
    home_wins = draws = away_wins = 0

    for _ in range(simulations):
        gh = np.random.poisson(lambda_home)
        ga = np.random.poisson(lambda_away)
        if gh > ga:
            home_wins += 1
        elif gh == ga:
            draws += 1
        else:
            away_wins += 1

    sim_prob_H = home_wins / simulations
    sim_prob_D = draws / simulations
    sim_prob_A = away_wins / simulations

    # Изчисляване на Edge
    edge_H = sim_prob_H - prob_H
    edge_D = sim_prob_D - prob_D
    edge_A = sim_prob_A - prob_A

    # --- HTML резултат ---
    html = f"<h2>💰 Value Bets за мача: {home} 🆚 {away}</h2><ul>"

    if edge_H > min_edge:
        html += f"<li>🏠 Победа {home}: Value {edge_H*100:.1f}% " \
                f"(реална {sim_prob_H*100:.1f}% vs импл. {prob_H*100:.1f}%)</li>"
    if edge_D > min_edge:
        html += f"<li>🤝 Равенство: Value {edge_D*100:.1f}% " \
                f"(реална {sim_prob_D*100:.1f}% vs импл. {prob_D*100:.1f}%)</li>"
    if edge_A > min_edge:
        html += f"<li>🚀 Победа {away}: Value {edge_A*100:.1f}% " \
                f"(реална {sim_prob_A*100:.1f}% vs импл. {prob_A*100:.1f}%)</li>"

    if html.endswith("<ul>"):
        html += "<li>Няма забележими value bets.</li>"

    html += "</ul>"
    return html

# --- Главна функция за Flask ---
def predict_match(home, away, data=None):
    if data is None:
        return "<p>DataFrame с мачове не е подаден.</p>"
    if home == away:
        return "<p>Моля, избери два различни отбора.</p>"
    return find_value_bets(home, away, data)

# --- Унифициран run() метод за всички анализатори ---
def run(home, away, data=None):
    return predict_match(home, away, data)

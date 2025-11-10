import numpy as np
from analyzers.data import team_strength, mean_home_goals, mean_away_goals

def run(home, away, simulations=3000):
    # --- извличаме фактори
    home_attack, home_defense, _ = team_strength(home, True)
    away_attack, away_defense, _ = team_strength(away, False)

    lambda_home = mean_home_goals * (home_attack / max(0.1, away_defense))
    lambda_away = mean_away_goals * (away_attack / max(0.1, home_defense))

    # симулация на мачове
    home_wins = draws = away_wins = 0
    score_counts = {}
    for _ in range(simulations):
        gh = np.random.poisson(lambda_home)
        ga = np.random.poisson(lambda_away)
        if gh > ga:
            home_wins += 1
        elif gh == ga:
            draws += 1
        else:
            away_wins += 1
        score_counts[(gh, ga)] = score_counts.get((gh, ga), 0) + 1

    prob_home = home_wins / simulations
    prob_draw = draws / simulations
    prob_away = away_wins / simulations

    # Изчисляване на шанс за успех (1-8)
    chance_score = int(4 + 4 * max(prob_home, prob_draw, prob_away))  # от 4 до 8

    # HTML output (без точен резултат)
    html = f"<h2>{home} vs {away}</h2>"
    html += "<p>Вероятности: 1 (домакин) <b>{:.0%}</b>, X (равен) <b>{:.0%}</b>, 2 (гост) <b>{:.0%}</b></p>".format(prob_home, prob_draw, prob_away)

    return html, (prob_home, prob_draw, prob_away), chance_score

import numpy as np
from scipy.stats import poisson
from analyzers.data import data, mean_home_goals, mean_away_goals, team_strength, historical_btts_rate

def run(home, away, simulations=3000):
    home_attack, home_defense, _ = team_strength(home, True)
    away_attack, away_defense, _ = team_strength(away, False)

    lambda_home = mean_home_goals * (home_attack / max(0.1, away_defense))
    lambda_away = mean_away_goals * (away_attack / max(0.1, home_defense))
    lambda_home = float(np.clip(lambda_home, 0.1, 5.0))
    lambda_away = float(np.clip(lambda_away, 0.1, 5.0))

    btts_sim = 0
    for _ in range(simulations):
        gh = np.random.poisson(lambda_home)
        ga = np.random.poisson(lambda_away)
        if gh > 0 and ga > 0:
            btts_sim += 1
    model_btts = btts_sim / simulations

    hist_btts = historical_btts_rate(home, away, last_matches=20)

    # комбиниране: доверие в модела и историческата честота
    alpha = 2.5
    beta = 1.5
    final_btts = (model_btts * alpha + hist_btts * beta) / (alpha + beta)

    html = f"<h2>{home} 🆚 {away}</h2>"
    html += "<p><b>⚡ Анализ: BTTS</b></p>"
    html += f"<p>Model BTTS: {model_btts*100:.1f}%</p>"
    html += f"<p>Historical BTTS (avg teams): {hist_btts*100:.1f}%</p>"
    html += f"<p>Final BTTS (blended): {final_btts*100:.1f}%</p>"

    return html

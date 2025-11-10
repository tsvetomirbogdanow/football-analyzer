import numpy as np
from analyzers.data import mean_home_goals, mean_away_goals, team_strength
from scipy.stats import poisson

def run(home, away, simulations=3000):
    # Симулация на мач и проверка за хендикап линии (-0.5, -1, +0.5 и т.н.)
    ha, hd, _ = team_strength(home, True)
    aa, ad, _ = team_strength(away, False)

    lambda_home = mean_home_goals * (ha / max(0.1, ad))
    lambda_away = mean_away_goals * (aa / max(0.1, hd))
    lambda_home = float(np.clip(lambda_home, 0.1, 5.0))
    lambda_away = float(np.clip(lambda_away, 0.1, 5.0))

    counts = {'home_win':0, 'draw':0, 'away_win':0,
              'home_cover_-0.5':0, 'home_cover_-1':0, 'away_cover_+0.5':0 }

    for _ in range(simulations):
        gh = np.random.poisson(lambda_home)
        ga = np.random.poisson(lambda_away)
        if gh>ga: counts['home_win'] +=1
        elif gh==ga: counts['draw'] +=1
        else: counts['away_win'] +=1

        # -0.5 (home needs strict lead)
        if gh - ga > 0:
            counts['home_cover_-0.5'] +=1
        # -1 (home needs win by >=2 for full cover; win by 1 is push in many lines)
        if gh - ga >= 2:
            counts['home_cover_-1'] +=1
        # +0.5 away cover (away not lose)
        if ga - gh >= 0:
            counts['away_cover_+0.5'] +=1

    total = simulations
    html = f"<h2>{home} 🆚 {away}</h2>"
    html += "<p><b>📉 Анализ: Хендикап (симулация)</b></p>"
    html += f"<p>Home win: {counts['home_win']/total*100:.1f}% | Draw: {counts['draw']/total*100:.1f}% | Away win: {counts['away_win']/total*100:.1f}%</p>"
    html += f"<p>Home covers -0.5: {counts['home_cover_-0.5']/total*100:.1f}%</p>"
    html += f"<p>Home covers -1 (win by 2+): {counts['home_cover_-1']/total*100:.1f}%</p>"
    html += f"<p>Away not lose (+0.5): {counts['away_cover_+0.5']/total*100:.1f}%</p>"

    return html

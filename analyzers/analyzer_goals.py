import numpy as np
from analyzers.data import data, mean_home_goals, mean_away_goals, team_strength, normalize_implied
from scipy.stats import poisson

def run(home, away, simulations=3000):
    home_attack, home_defense, _ = team_strength(home, True)
    away_attack, away_defense, _ = team_strength(away, False)

    lambda_home = mean_home_goals * (home_attack / max(0.1, away_defense))
    lambda_away = mean_away_goals * (away_attack / max(0.1, home_defense))
    lambda_home = float(np.clip(lambda_home, 0.1, 5.0))
    lambda_away = float(np.clip(lambda_away, 0.1, 5.0))

    # моделна probability for total goals > k can be computed exactly using Poisson convolution,
    # but for simplicity и consistency с останалата част използваме симулация
    totals = []
    for _ in range(simulations):
        gh = np.random.poisson(lambda_home)
        ga = np.random.poisson(lambda_away)
        totals.append(gh+ga)

    over_0_5 = sum(1 for t in totals if t > 0) / len(totals)
    over_1_5 = sum(1 for t in totals if t > 1) / len(totals)
    over_2_5 = sum(1 for t in totals if t > 2) / len(totals)
    over_3_5 = sum(1 for t in totals if t > 3) / len(totals)

    # използваме пазарните Avg>2.5 / Avg<2.5 за корекция ако налични
    if 'Avg>2.5' in data.columns and 'Avg<2.5' in data.columns:
        # Взимаме първия ред просто да прочетем пазарния average (позицията не важи, само колоните)
        row = data.iloc[0]
        try:
            odd_over = float(row['Avg>2.5'])
            odd_under = float(row['Avg<2.5'])
            imp_over = 1.0/odd_over
            imp_under = 1.0/odd_under
            s = imp_over + imp_under
            if s>0:
                imp_p_over = imp_over / s
                # комбинираме model and implied (тежести)
                alpha = 3.0
                beta = 1.0
                final_over_2_5 = (over_2_5 * alpha + imp_p_over * beta) / (alpha + beta)
            else:
                final_over_2_5 = over_2_5
        except:
            final_over_2_5 = over_2_5
    else:
        final_over_2_5 = over_2_5

    html = f"<h2>{home} 🆚 {away}</h2>"
    html += "<p><b>⚽ Анализ: Брой голове (Over/Under)</b></p>"
    html += f"<p>Over 0.5: {over_0_5*100:.1f}%</p>"
    html += f"<p>Over 1.5: {over_1_5*100:.1f}%</p>"
    html += f"<p>Over 2.5 (model): {over_2_5*100:.1f}%</p>"
    html += f"<p>Over 2.5 (final): {final_over_2_5*100:.1f}%</p>"
    html += f"<p>Over 3.5: {over_3_5*100:.1f}%</p>"

    return html

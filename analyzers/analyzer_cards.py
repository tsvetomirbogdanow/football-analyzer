import numpy as np
from analyzers.data import data, league_yellow, weighted_avg

def run(home, away, last_matches=10, simulations=1000):
    # Проверка за налични колони
    required_cols = ['HY','AY','HF','AF']
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        return f"<p>⚠️ CSV файловете нямат необходимите колони: {', '.join(missing_cols)}</p>"

    # Исторически данни за домакин и гост
    df_home = data[(data['HomeTeam']==home) | (data['AwayTeam']==home)]
    df_away = data[(data['HomeTeam']==away) | (data['AwayTeam']==away)]

    # Средни жълти картони
    home_y = weighted_avg(df_home[df_home['HomeTeam']==home], 'HY', last_matches)
    away_y = weighted_avg(df_away[df_away['AwayTeam']==away], 'AY', last_matches)

    # Средни фолове
    home_fouls = weighted_avg(df_home[df_home['HomeTeam']==home], 'HF', last_matches)
    away_fouls = weighted_avg(df_away[df_away['AwayTeam']==away], 'AF', last_matches)

    # Ако няма данни, използваме лигова средна или фиктивни стойности
    home_y = home_y if not np.isnan(home_y) else (league_yellow/2 if league_yellow else 1.5)
    away_y = away_y if not np.isnan(away_y) else (league_yellow/2 if league_yellow else 1.5)
    home_fouls = home_fouls if not np.isnan(home_fouls) else 10
    away_fouls = away_fouls if not np.isnan(away_fouls) else 10

    # Комбиниране на метриките за Poisson λ
    alpha, beta = 0.7, 0.3
    lambda_home = alpha*home_y + beta*(home_fouls/5)
    lambda_away = alpha*away_y + beta*(away_fouls/5)

    # Poisson симулации
    samples_home = np.random.poisson(max(0.2, lambda_home), simulations)
    samples_away = np.random.poisson(max(0.2, lambda_away), simulations)
    samples_total = samples_home + samples_away

    # Over X за всеки отбор
    overs_home = {f"Over {x}.5": np.mean(samples_home > x)*100 for x in [1,2,3,4,5]}
    overs_away = {f"Over {x}.5": np.mean(samples_away > x)*100 for x in [1,2,3,4,5]}
    overs_total = {f"Over {x}.5": np.mean(samples_total > x)*100 for x in [1,2,3,4,5]}

    # HTML резултат
    html = f"<h2>{home} 🆚 {away}</h2>"
    html += "<p><b>🟨 Анализ: Жълти картони и дисциплина</b></p>"

    html += f"<p><b>{home} - вероятности за Over:</b></p><ul>"
    for k,v in overs_home.items():
        html += f"<li>{k}: {v:.1f}%</li>"
    html += "</ul>"

    html += f"<p><b>{away} - вероятности за Over:</b></p><ul>"
    for k,v in overs_away.items():
        html += f"<li>{k}: {v:.1f}%</li>"
    html += "</ul>"

    html += f"<p><b>Общо картони - вероятности за Over:</b></p><ul>"
    for k,v in overs_total.items():
        html += f"<li>{k}: {v:.1f}%</li>"
    html += "</ul>"

    return html

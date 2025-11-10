import numpy as np
from analyzers.data import data, league_corners, weighted_avg

def run(home, away, last_matches=10, simulations=2000):
    # Проверка за нужните колони
    if not all(c in data.columns for c in ['HC','AC']):
        return "<p>⚠️ CSV файловете нямат колони 'HC' и 'AC' за корнери.</p>"

    # Исторически данни
    df_home = data[(data['HomeTeam']==home) | (data['AwayTeam']==home)]
    df_away = data[(data['HomeTeam']==away) | (data['AwayTeam']==away)]

    # Средни корнери
    h_for = weighted_avg(df_home[df_home['HomeTeam']==home], 'HC', last_matches)
    h_against = weighted_avg(df_home[df_home['AwayTeam']==home], 'AC', last_matches)
    a_for = weighted_avg(df_away[df_away['AwayTeam']==away], 'AC', last_matches)
    a_against = weighted_avg(df_away[df_away['HomeTeam']==away], 'HC', last_matches)

    # Ако няма данни – използваме средна за лигата
    if np.isnan(h_for): h_for = league_corners/2 if league_corners else 5.0
    if np.isnan(a_for): a_for = league_corners/2 if league_corners else 5.0
    if np.isnan(h_against): h_against = league_corners/2 if league_corners else 5.0
    if np.isnan(a_against): a_against = league_corners/2 if league_corners else 5.0

    # Очаквани корнери
    exp_home = (h_for + a_against)/2
    exp_away = (a_for + h_against)/2
    exp_total = exp_home + exp_away

    # Симулиране Poisson
    samples_home = np.random.poisson(max(0.5, exp_home), simulations)
    samples_away = np.random.poisson(max(0.5, exp_away), simulations)
    samples_total = samples_home + samples_away

    # Over thresholds
    thresholds_individual = [3.5, 5.5, 7.5]  # за домакин и гост
    thresholds_total = [8.5, 9.5, 11.5, 12.5]  # за общи корнери

    probs_home = [np.mean(samples_home > t)*100 for t in thresholds_individual]
    probs_away = [np.mean(samples_away > t)*100 for t in thresholds_individual]
    probs_total = [np.mean(samples_total > t)*100 for t in thresholds_total]

    # HTML изход
    html = f"<h2>{home} 🆚 {away}</h2>"
    html += "<p><b>🚩 Анализ: Корнери</b></p>"

    # Отделно за домакин
    html += "<p>Вероятности Over X (домакин):</p>"
    for t, p in zip(thresholds_individual, probs_home):
        html += f"<p>Over {t}: {p:.1f}%</p>"

    # Отделно за гост
    html += "<p>Вероятности Over X (гост):</p>"
    for t, p in zip(thresholds_individual, probs_away):
        html += f"<p>Over {t}: {p:.1f}%</p>"

    # Общи корнери
    html += "<p><b>Общо очаквани корнери:</b></p>"
    for t, p in zip(thresholds_total, probs_total):
        html += f"<p>Over {t}: {p:.1f}%</p>"

    return html

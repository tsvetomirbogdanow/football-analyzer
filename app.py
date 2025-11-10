from flask import Flask, render_template, request
from analyzers import analyzer_1x2
from analyzers.data import teams

app = Flask(__name__)

# --- Basic Auth ---
USERS = {"client1": "password1", "client2": "password2"}

def check_auth(username, password):
    return USERS.get(username) == password

def authenticate():
    from flask import Response
    return Response('Неуспешен достъп.', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})


@app.before_request
def require_auth():
    from flask import request
    auth = request.authorization
    if not auth or not check_auth(auth.username, auth.password):
        return authenticate()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    chance = 7
    prob_home = prob_draw = prob_away = None
    selected_home = selected_away = None
    predicted_result = None

    if request.method == "POST":
        selected_home = request.form.get("home_team")
        selected_away = request.form.get("away_team")

        if selected_home and selected_away and selected_home != selected_away:
            # Взимаме анализа
            html_result, probs, chance_score = analyzer_1x2.run(selected_home, selected_away)
            result = html_result
            prob_home, prob_draw, prob_away = probs
            chance = chance_score

            # --- Конвертиране в проценти, ако са в 0..1 ---
            if max(probs) <= 1.0:
                prob_home *= 100
                prob_draw *= 100
                prob_away *= 100

            probs = [prob_home, prob_draw, prob_away]

            # Изчисляваме максимална, средна и минимална вероятност
            max_prob = max(probs)
            min_prob = min(probs)
            second_prob = sorted(probs, reverse=True)[1]

            outcomes = [("Домакин", prob_home), ("Равенство", prob_draw), ("Гост", prob_away)]
            outcomes_sorted = sorted(outcomes, key=lambda x: x[1], reverse=True)

            # 1️⃣ Твърде рисков за залог
            if max_prob - min_prob <= 10 and max_prob < 50:
                predicted_result = "Твърде рисков за залог"
                chance = 2

            # 2️⃣ Двоен шанс
            elif max_prob - second_prob <= 15:
                top_two = outcomes_sorted[:2]
                top_names = [x[0] for x in top_two]

                if "Домакин" in top_names and "Равенство" in top_names:
                    predicted_result = f"Двоен шанс 1X ({selected_home} или Равен)"
                elif "Гост" in top_names and "Равенство" in top_names:
                    predicted_result = f"Двоен шанс X2 (Равен или {selected_away})"
                elif "Домакин" in top_names and "Гост" in top_names:
                    predicted_result = f"Двоен шанс 12 ({selected_home} или {selected_away})"
                else:
                    predicted_result = f"{selected_home} победа"

            # 3️⃣ Най-високата вероятност
            else:
                if outcomes_sorted[0][0] == "Домакин":
                    predicted_result = f"{selected_home} победа"
                elif outcomes_sorted[0][0] == "Гост":
                    predicted_result = f"{selected_away} победа"
                else:
                    predicted_result = "Равенство"

    return render_template(
        "index.html",
        teams=teams,
        result=result,
        selected_home=selected_home,
        selected_away=selected_away,
        prob_home=prob_home,
        prob_draw=prob_draw,
        prob_away=prob_away,
        chance=chance,
        predicted_result=predicted_result
    )


if __name__ == "__main__":
    app.run(debug=True, port=8080)

from flask import Flask, render_template, request
from flask_wtf import CSRFProtect
from flask_csp.csp import csp_header
import logging
import pickle
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

app_log = logging.getLogger(__name__)
logging.basicConfig(
    filename="security_log.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)

app = Flask(__name__)
app.secret_key = b"_53oi3uriq9pifpff;apl"
csrf = CSRFProtect(app)

filename = "my_saved_model.sav"
loaded_model = pickle.load(open(filename, "rb"))

# Approximate max values for scaling to -1 -> 1
# for GOLD_MAX the approximate
GOLD_MAX = 20000
XP_MAX = 15000


def scale_to_range(value, max_value):
    """Scale a difference value to -1 -> 1 range."""
    scaled = value / max_value
    return max(-1, min(1, scaled))


def calculate_gold_xp_advantage(gold_diff, xp_diff):
    """Convert raw gold and xp differences to combined GoldXpAdvantage."""
    scaled_gold = scale_to_range(gold_diff, GOLD_MAX)
    scaled_xp = scale_to_range(xp_diff, XP_MAX)
    return (scaled_gold + scaled_xp) / 2


def generate_graph(user_value=None, user_prediction=None):
    x = np.linspace(-1, 1, 300).reshape(-1, 1)
    probabilities = loaded_model.predict_proba(x)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, probabilities, color="royalblue", linewidth=2, label="P(Blue Wins)")
    ax.axhline(
        0.5, color="gray", linestyle="--", linewidth=1, label="Decision boundary (0.5)"
    )

    if user_value is not None:
        ax.axvline(
            user_value,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"Your input ({user_value:.3f})",
        )
        ax.scatter(
            [user_value],
            [user_prediction],
            color="red",
            zorder=5,
            s=80,
            label=f"Win probability ({user_prediction:.2f})",
        )

    ax.set_xlabel("Gold & XP Advantage (-1 to 1)")
    ax.set_ylabel("Probability of Blue Win")
    ax.set_title("Sigmoid Classification: Blue Win Probability")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return graph_base64


@app.route("/", methods=["GET", "POST"])
@csp_header(
    {
        "base-uri": "'self'",
        "default-src": "'self'",
        "style-src": "'self'",
        "script-src": "'self'",
        "img-src": "'self' data:",
        "media-src": "'self'",
        "font-src": "'self'",
        "object-src": "'self'",
        "child-src": "'self'",
        "connect-src": "'self'",
        "worker-src": "'self'",
        "report-uri": "/csp_report",
        "frame-ancestors": "'none'",
        "form-action": "'self'",
        "frame-src": "'none'",
    }
)
def index():
    prediction = None
    graph = generate_graph()
    scaled_value = None

    if request.method == "POST":
        blue_gold = request.form.get("blue_gold")
        red_gold = request.form.get("red_gold")
        blue_xp = request.form.get("blue_xp")
        red_xp = request.form.get("red_xp")

        if not all([blue_gold, red_gold, blue_xp, red_xp]):
            return render_template(
                "index.html",
                prediction="Please fill in all fields.",
                graph=graph,
                scaled_value=None,
            )

        try:
            gold_diff = float(blue_gold) - float(red_gold)
            xp_diff = float(blue_xp) - float(red_xp)

            scaled_value = calculate_gold_xp_advantage(gold_diff, xp_diff)

            input_data = np.array([[scaled_value]])
            result = loaded_model.predict(input_data)[0]
            prob = loaded_model.predict_proba(input_data)[0][1]
            prediction = "Blue Wins" if result == 1 else "Blue Loses"
            win_percent = round(prob * 100, 1)
            graph = generate_graph(user_value=scaled_value, user_prediction=prob)
        except ValueError:
            return render_template(
                "index.html",
                prediction="Please enter valid numbers.",
                graph=graph,
                scaled_value=None,
                win_percent=None,
            )

    return render_template(
        "index.html",
        prediction=prediction,
        graph=graph,
        scaled_value=scaled_value,
        win_percent=win_percent if prediction else None,
    )


@app.route("/offline.html", methods=["GET"])
def offline():
    return render_template("offline.html")


@app.route("/csp_report", methods=["POST"])
@csrf.exempt
def csp_report():
    app.logger.critical(request.data.decode())
    return "done"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)

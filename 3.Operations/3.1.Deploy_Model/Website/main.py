from flask import Flask, render_template, request, redirect, url_for, session
from flask_wtf import CSRFProtect
from flask_csp.csp import csp_header
from functools import wraps
import logging
import pickle
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import userManagement as dbHandler

app_log = logging.getLogger(__name__)
logging.basicConfig(
    filename="security_log.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = b"_53oi3uriq9pifpff;apl"
csrf = CSRFProtect(app)

# ─── Model Loading ────────────────────────────────────────────────────────────
filename = "my_saved_model.sav"
loaded_model = pickle.load(open(filename, "rb"))

GOLD_MAX = 30000
XP_MAX = 15000

# ─── CSP Policy ───────────────────────────────────────────────────────────────
CSP_POLICY = {
    "base-uri": "'self'",
    "default-src": "'self'",
    "style-src": "'self'",
    "script-src": "'self'",
    "img-src": "'self' data:",
    "media-src": "'self'",
    "font-src": "'self'",
    "object-src": "'none'",
    "child-src": "'self'",
    "connect-src": "'self'",
    "worker-src": "'self'",
    "report-uri": "/csp_report",
    "frame-ancestors": "'none'",
    "form-action": "'self'",
    "frame-src": "'none'",
}

# ─── DB Init ──────────────────────────────────────────────────────────────────
try:
    dbHandler.getUsers()
    dbHandler.init_2fa_column()
except Exception as e:
    app_log.error("Failed to initialize database: %s", e)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated_function


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


# ─── Auth Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def root():
    return redirect(url_for("login"))


@app.route("/Login.html", methods=["GET", "POST"])
@csp_header(CSP_POLICY)
def login():
    if "user" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            app_log.warning("Login attempt with missing credentials.")
            return render_template(
                "Login.html",
                message="Please enter both username and password.",
                message_type="danger",
            )

        if dbHandler.authenticate(username, password):
            session["pre_2fa_user"] = username
            app_log.info("User %s passed password check, redirecting to 2FA.", username)
            return redirect(url_for("two_factor"))
        else:
            app_log.warning("Failed login attempt for user: %s", username)
            return render_template(
                "Login.html",
                message="Invalid username or password.",
                message_type="danger",
            )

    return render_template("Login.html")


@app.route("/Signup.html", methods=["GET", "POST"])
@csp_header(CSP_POLICY)
def signup():
    if "user" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        password_confirm = request.form.get("password_confirm")

        if not email or not password or not password_confirm:
            return render_template("Signup.html", message="All fields are required")

        if password != password_confirm:
            return render_template("Signup.html", message="Passwords do not match")

        success, msg = dbHandler.NewUser(email, password)
        if not success:
            return render_template(
                "Signup.html",
                message=msg,
                message_type="danger",
            )

        app_log.info("New user registered: %s", email)
        return redirect(url_for("login"))

    return render_template("Signup.html")


@app.route("/2fa.html", methods=["GET", "POST"])
@csp_header(CSP_POLICY)
def two_factor():
    username = session.get("pre_2fa_user")
    if not username:
        return redirect(url_for("login"))

    qr_code = dbHandler.get_2fa_qr_code_base64(username)
    secret_key = dbHandler.get_2fa_key(username)

    if request.method == "POST":
        code = request.form.get("code", "").strip()
        if dbHandler.verify_2fa_code(username, code):
            session.pop("pre_2fa_user", None)
            session["user"] = username
            app_log.info("User %s completed 2FA successfully.", username)
            return redirect(url_for("index"))
        else:
            app_log.warning("Failed 2FA attempt for user: %s", username)
            return render_template(
                "2fa.html",
                message="Invalid code. Please try again.",
                message_type="danger",
                qr_code=qr_code,
                secret_key=secret_key,
            )

    return render_template("2fa.html", qr_code=qr_code, secret_key=secret_key)


@app.route("/logout")
def logout():
    session.clear()
    app_log.info("User logged out.")
    return redirect(url_for("login"))


# ─── Main Predictor Route ─────────────────────────────────────────────────────
@app.route("/index.html", methods=["GET", "POST"])
@csp_header(CSP_POLICY)
@login_required
def index():
    prediction = None
    graph = generate_graph()
    scaled_value = None
    win_percent = None
    error = None

    if request.method == "POST":
        blue_gold = request.form.get("blue_gold")
        red_gold = request.form.get("red_gold")
        blue_xp = request.form.get("blue_xp")
        red_xp = request.form.get("red_xp")

        if not all([blue_gold, red_gold, blue_xp, red_xp]):
            error = "Please fill in all fields."
        else:
            try:
                blue_gold_f = float(blue_gold)
                red_gold_f = float(red_gold)
                blue_xp_f = float(blue_xp)
                red_xp_f = float(red_xp)

                if any(v < 0 for v in [blue_gold_f, red_gold_f, blue_xp_f, red_xp_f]):
                    error = "Gold and XP values must be non-negative."

                elif blue_gold_f > GOLD_MAX or red_gold_f > GOLD_MAX:
                    error = f"Gold values cannot exceed {GOLD_MAX}."

                elif blue_xp_f > XP_MAX or red_xp_f > XP_MAX:
                    error = f"XP values cannot exceed {XP_MAX}."

                else:
                    gold_diff = blue_gold_f - red_gold_f
                    xp_diff = blue_xp_f - red_xp_f
                    scaled_value = calculate_gold_xp_advantage(gold_diff, xp_diff)

                    input_data = np.array([[scaled_value]])
                    result = loaded_model.predict(input_data)[0]
                    prob = loaded_model.predict_proba(input_data)[0][1]
                    prediction = "Blue Wins" if result == 1 else "Blue Loses"
                    win_percent = round(prob * 100, 1)
                    graph = generate_graph(
                        user_value=scaled_value, user_prediction=prob
                    )

            except ValueError:
                error = "Please enter valid numbers."

    return render_template(
        "index.html",
        prediction=prediction,
        graph=graph,
        scaled_value=scaled_value,
        win_percent=win_percent,
        error=error,
    )


# ─── Utility Routes ───────────────────────────────────────────────────────────
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

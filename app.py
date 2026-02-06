import os
from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


# -----------------------------
# App Configuration
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

# -----------------------------
# Dataset (Numeric Target for Regression)
# -----------------------------

# Target now represents "Pizza Desire %" (0–100)
data = {
    'Am_I_Hungry': [1, 1, 1, 0, 0],
    'Is_It_Weekend': [1, 0, 1, 1, 0],
    'Pizza_Score': [90, 75, 95, 40, 10]   # Continuous values ✅
}

df = pd.DataFrame(data)

# Features & Target
X = df[['Am_I_Hungry', 'Is_It_Weekend']]
y = df['Pizza_Score']

# -----------------------------
# Train Regression Model
# -----------------------------

model = DecisionTreeRegressor(
    random_state=42,
    max_depth=3
)

model.fit(X, y)

# -----------------------------
# Routes
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":

        hungry = request.form.get("hungry")
        weekend = request.form.get("weekend")

        # Convert input
        hungry_val = 1 if hungry == "Yes" else 0
        weekend_val = 1 if weekend == "Yes" else 0

        # Predict
        test = pd.DataFrame(
            [[hungry_val, weekend_val]],
            columns=["Am_I_Hungry", "Is_It_Weekend"]
        )

        pred = model.predict(test)[0]

        result = f"Pizza Desire Score: {pred:.1f}% 🍕"

    return render_template("index.html", result=result)

# -----------------------------
# Run App (Local + Render)
# -----------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)

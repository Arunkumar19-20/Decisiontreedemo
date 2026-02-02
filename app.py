import os
from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Get current folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Tell Flask exactly where templates folder is
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

# -----------------------------
# Dataset
# -----------------------------
data = {
    'Am_I_Hungry': ['Yes', 'Yes', 'Yes', 'No', 'No'],
    'Is_It_Weekend': ['Yes', 'No', 'Yes', 'Yes', 'No'],
    'Shall_I_Eat_Pizza': ['Yes', 'Yes', 'Yes', 'No', 'No']
}

df = pd.DataFrame(data)

# Encode Yes/No → 1/0
df = df.replace({'Yes': 1, 'No': 0})

X = df[['Am_I_Hungry', 'Is_It_Weekend']]
y = df['Shall_I_Eat_Pizza']

# Train model
model = DecisionTreeClassifier(
    criterion='entropy',
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

        if pred == 1:
            result = "Yes, You Should Eat Pizza 🍕"
        else:
            result = "No, Better Avoid Pizza ❌"

    return render_template("index.html", result=result)


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run()
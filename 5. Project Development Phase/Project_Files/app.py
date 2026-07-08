from flask import Flask, render_template, request
import joblib
import os
app = Flask(__name__)

# Load trained model
model = joblib.load("model/model.pkl")

# Crop images
crop_images = {
    "apple": "apple.jpg",
    "banana": "banana.jpg",
    "blackgram": "blackgram.jpg",
    "chickpea": "chickpea.jpg",
    "coconut": "coconut.jpg",
    "coffee": "coffee.jpg",
    "cotton": "cotton.jpg",
    "grapes": "grapes.jpg",
    "jute": "jute.jpg",
    "kidneybeans": "kidneybeans.jpg",
    "lentil": "lentil.jpg",
    "maize": "maize.jpg",
    "mango": "mango.jpg",
    "mothbeans": "mothbeans.jpg",
    "mungbean": "mungbean.jpg",
    "muskmelon": "muskmelon.jpg",
    "orange": "orange.jpg",
    "papaya": "papaya.jpg",
    "pigeonpeas": "pigeonpeas.jpg",
    "pomegranate": "pomegranate.jpg",
    "rice": "rice.jpg",
    "watermelon": "watermelon.jpg"
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/recommend")
def recommend():
    return render_template("recommend.html")


@app.route("/predict", methods=["POST"])
def predict():

    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])

    crop = prediction[0].lower()

    image = crop_images.get(crop, "default.jpg")

    return render_template(
        "result.html",
        crop=prediction[0],
        image=image
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)    
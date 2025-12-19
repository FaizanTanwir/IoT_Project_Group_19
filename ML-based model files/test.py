from machine import Pin
import machine
from machine import Pin, I2C
import bme280
import time


i2c = I2C(0, sda=Pin(0), scl=Pin(1), freq=400000)  # id=channel
bme = bme280.BME280(i2c=i2c)
try:
    from pressure_comfort_model import RandomForestClassifier
    clf = RandomForestClassifier()
    ML_AVAILABLE = True
    print("[INFO] ML model loaded successfully (class)!")
except ImportError:
    ML_AVAILABLE = False
    print("[WARNING] ML model not found, using rule-based logic")


def predict_fallback(temp, hum, press):
    """Fallback rule-based prediction"""
    if 20 <= temp <= 25 and 30 <= hum <= 60:
        return "Comfort", 0.80
    else:
        return "Uncomf", 0.80


def predict_ml(temp, hum, press):
    """Use ML model for prediction"""
    try:
        # Try module-level `predict` first (if generated that way)
        try:
            prediction = predict([temp, hum, press])
        except NameError:
            # Otherwise use the classifier instance `clf` (if present).
            # The generated RandomForest expects pressure in Pascals (~101700),
            # while the code passes hPa (around 1000); convert when it looks like hPa.
            p = press * 100 if press < 2000 else press
            prediction = clf.predict([temp, hum, p])

        # prediction is 0 or 1
        if prediction == 1:
            return "Comfort", 0.95
        else:
            return "Uncomf", 0.95
    except Exception as e:
        print(f"[ERROR] ML prediction failed: {e}")
        return predict_fallback(temp, hum, press)


def predict_comfort(temp, hum, press):
    """Main prediction function - uses ML if available"""
    if ML_AVAILABLE:
        return predict_ml(temp, hum, press)
    else:
        return predict_fallback(temp, hum, press)


model_type = "ML" if ML_AVAILABLE else "Rules"
print(f"\n=== Starting with {model_type} prediction ===\n")

while True:
    try:
        # Read sensor
        
        temperature, pressure, humidity = bme.read_compensated_data()
        pressure_hpa = pressure / 100

        # Predict using ML or fallback
        status, confidence = predict_comfort(
            temperature, humidity, pressure_hpa)

        # Serial output
        print(
            f"Temp: {temperature:.1f}°C, Hum: {humidity:.1f}%, Press: {pressure_hpa:.1f}hPa")
        print(
            f"Prediction ({model_type}): {status} ({confidence*100:.0f}% confidence)\n")

        time.sleep(2)

    except Exception as e:
        print(f"[ERROR] {e}")
        time.sleep(5)

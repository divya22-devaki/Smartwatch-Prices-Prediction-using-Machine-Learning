import pickle
from flask import Flask, render_template, request
import numpy as np

# Load the model
model1 = pickle.load(open('SW.pkl', 'rb'))

# Initialize the Flask application
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict')
def predict():
    return render_template('predict.html')

# Data prediction route
@app.route('/data_predict', methods=['POST'])
def data_predict():
    if request.method == 'POST':
        # Get form data
        Brand = request.form['Brand']
        Model = request.form['Model']
        Operating_System = request.form['Operating_System']
        Connectivity = request.form['Connectivity']
        Display_Type = request.form['Display_Type']
        Display_Size = request.form['Display_Size']
        Resolution = request.form['Resolution']
        Water_Resistance = request.form['Water_Resistance']
        Battery_Life = request.form['Battery_Life']
        GPS = request.form['GPS']
        NFC = request.form['NFC']
        
        # Encode categorical variables
        Brand_dict = {'Garmin': 8, 'Mobvoi': 18, 'Fitbit': 6, 'Fossil': 7, 'Amazfit': 0, 'Samsung': 30, 'Huawei': 10, 'TicWatch': 35, 'Xiamoi': 36, 'Skagen': 31, 'Suunto': 33, 'Honor': 9, 'Apple': 1, 'Polar': 27, 'Casio': 3, 'Withings': 38, 'Oppo': 26, 'Timex': 37, 'Diesel': 4, 'Misfit': 17, 'Michael Kors': 16, 'Zepp': 41, 'LG': 13, 'TAG Heuer': 34, 'Asus': 2, 'Montblanc': 19, 'Sony': 32, 'Realme': 29, 'Matrix': 15, 'Kate Spade': 11, 'Kospet': 12, 'Emporio Armani': 5, 'Nokia': 23, 'MyKronoz': 22, 'Zeblaze': 40, 'Lemfo': 14, 'Nubia': 24, 'Moto': 20, 'Polaroid': 28, 'Motorola': 21, 'OnePlus': 25}
        Model_dict = {'7': 0, '9 Baro': 1, '9 Peak': 2, 'Access Bradshaw 2': 3, 'Access Gen 5': 4, 'Access Runway': 5, 'Alpha': 6, 'Bip S': 7, 'Bip U Pro': 8, 'C2': 9, 'C2+': 10}
        Operating_System_dict = {'Amazfit OS': 0, 'Android': 1, 'Android OS': 2, 'Android Wear': 3, 'Casio OS': 4, 'ColorOS': 5, 'Custom OS': 6, 'Fitbit OS': 7, 'Fossil OS': 8, 'Garmin OS': 9, 'HarmonyOS': 10, 'Hybrid OS': 11, 'Lite OS': 12, 'LiteOS': 13, 'MIUI': 14, 'MIUI For Watch': 15, 'Matrix OS': 17, 'Mi Wear OS': 18, 'MyKronoz OS': 19, 'Nubia OS': 20, 'Polar OS': 21, 'Proprietary': 22, 'Proprietary OS': 23, 'RTOS': 24, 'Realme OS': 25, 'Skagen OS': 26, 'Suunto OS': 27, 'Timex OS': 28, 'Tizen': 29, 'Tizen OS': 30, 'Wear OS': 31, 'Withings OS': 32, 'Zepp OS': 33, 'watchOS': 34}
        Connectivity_dict = {'Bluetooth, Wi-Fi': 1, 'Bluetooth, Wi-Fi, Cellular': 2, 'Bluetooth': 0, 'Bluetooth, Wi-Fi, GPS': 3, 'Bluetooth, Wi-Fi, NFC': 4}
        Display_Type_dict = {'AMOLED': 0, 'Analog': 1, 'Color Touch': 2, 'Dual Layer': 3, 'E-Ink': 4, 'E-ink': 5, 'Gorilla Glass': 6, 'IPS': 7, 'IPS LCD': 8, 'LCD': 9, 'MIP': 10, 'Memory LCD': 11, 'Memory-in-pixel (MIP)': 12, 'Monochrome': 13, 'OLED': 14, 'P-OLED': 15, 'PMOLED': 16, 'Retina': 17, 'STN LCD': 18, 'Sunlight-visible': 19, 'Sunlight-visible, transflective memory-in-pixel (MIP)': 20, 'Super AMOLED': 21, 'TFT': 22, 'TFT LCD': 23, 'TFT-LCD': 24, 'Transflective': 25, 'transflective': 26}
        Display_Size_dict = {0.9: 0, 1.0: 1, 1.04: 2, 1.06: 3, 1.1: 4, 1.19: 5, 1.2: 6, 1.22: 7, 1.23: 8, 1.28: 9, 1.3: 10, 1.32: 11, 1.34: 12, 1.35: 13, 1.36: 14, 1.363164893617021: 15, 1.38: 16, 1.39: 17, 1.4: 18, 1.42: 19, 1.43: 20, 1.5: 21, 1.57: 22, 1.58: 23, 1.6: 24, 1.64: 25, 1.65: 26, 1.75: 27, 1.78: 28, 1.9: 29, 1.91: 30, 2.1: 31, 4.01: 32}
        Resolution_dict = {'126 x 36': 0, '128 x 128': 1, '160 x 160': 2, '176 x 176': 3, '200 x 200': 4, '228 x 172': 5, '240 x 198': 6, '240 x 201': 7, '240 x 240': 8, '260 x 260': 9, '280 x 280': 10, '280 x 456': 11, '300 x 300': 12, '320 x 300': 13, '320 x 302': 14, '320 x 320': 15, '324 x 394': 16, '326 x 326': 17, '328 x 328': 18, '336 x 336': 19, '348 x 250': 20, '348 x 442': 21, '360 x 360': 22, '368 x 448': 23, '372 x 430': 24, '390 x 390': 25, '394 x 324': 26, '396 x 484': 27, '400 x 400': 28, '402 x 476': 29, '416 x 416': 30, '450 x 450': 31, '454 x 454': 32, '466 x 466': 33, '480 x 480': 34, '960 x 192': 35}
        Water_Resistance_dict = {50: 5, 30: 4, 100: 2, 200: 3, 1.5: 0, 'Not Specified': 6, 10: 1}
        Battery_Life_dict = {'1': 0, '1.5': 1, '10': 2, '11': 3, '12': 4, '14': 5, '15': 6, '16': 7, '18': 8, '2': 9, '20': 10, '21': 11, '22': 12, '24': 13, '26': 14, '28': 15, '3': 16, '4': 17, '5': 18, '6': 19, '7': 20, '72': 21, '8': 22, '9': 23, 'Not Specified': 24}
        
        # Convert form data to numerical values
        Brand = Brand_dict.get(Brand, -1)  # Use -1 for unknown brands
        Model = Model_dict.get(Model, -1)  # Use -1 for unknown models
        Operating_System = Operating_System_dict.get(Operating_System, -1)  # Use -1 for unknown OS
        Connectivity = Connectivity_dict.get(Connectivity, -1)  # Use -1 for unknown connectivity
        Display_Type = Display_Type_dict.get(Display_Type, -1)  # Use -1 for unknown display type
        Display_Size = Display_Size_dict.get(float(Display_Size), -1)  # Use -1 for unknown sizes
        Resolution = Resolution_dict.get(Resolution, -1)  # Use -1 for unknown resolutions
        Water_Resistance = Water_Resistance_dict.get(float(Water_Resistance), -1)  # Use -1 for unknown water resistance
        Battery_Life = Battery_Life_dict.get(Battery_Life, -1)  # Use -1 for unknown battery life
        GPS = 1 if request.form['GPS'] == 'Yes' else 0
        NFC = 1 if request.form['NFC'] == 'Yes' else 0
        
        # Ensure all inputs are numerical
        features = [Brand, Model, Operating_System, Connectivity, Display_Type, Display_Size, Resolution, Water_Resistance, Battery_Life, GPS, NFC]

        # Ensure the feature list is of correct length
        if len(features) != 11:
            return render_template('watch_prediction.html', prediction_text='Feature shape mismatch.')

        # Make prediction
        prediction = model1.predict([features])
        output = round(prediction[0], 2)

        return render_template('watch_prediction.html', prediction_text=f'The Price of Smartwatch is : ${output}')

if __name__ == "__main__":
    app.run(debug=True)

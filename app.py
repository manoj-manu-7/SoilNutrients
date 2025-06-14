from flask import Flask, render_template, request
import pickle
import numpy as np
import os

filename = 'xg_model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form inputs safely with validation
            ph = request.form.get('ph', '').strip()  # Get and strip whitespace
            if not ph:  # Check if empty
                return "Error: pH value is missing!", 400

            ph = float(ph)  # Convert to float
            
            # Get other form inputs with default values to prevent crashes
            Organic = float(request.form.get('Organic', 0))
            CEC = float(request.form.get('CEC', 0))
            temp = float(request.form.get('temp', 0))
            Precipitation = float(request.form.get('Precipitation', 0))
            soiltype = int(request.form.get('soiltype', 0))
            Texture = int(request.form.get('Texture', 0))
            land = int(request.form.get('land', 0))
            modelSelected = int(request.form.get('selectModel', 0))

            # Process the prediction
            sample_input = np.array([[soiltype, ph, Organic, CEC, Texture, temp, Precipitation, land]])
            filedata = ['dt_model.pkl', 'knn_model.pkl', 'mlp_model.pkl', 'rf_model.pkl', 'xg_model.pkl']
            filename = filedata[modelSelected]
            model = pickle.load(open(filename, 'rb'))
            my_prediction = model.predict(sample_input)
            prediction_list = my_prediction.tolist()

            # Define suitability conditions
            suitable = False  # Default assumption
            if (ph >= 6.0 and ph <= 7.5) and (Organic >= 1.5) and (CEC >= 10) and (temp >= 15 and temp <= 35):
                suitable = True  # If conditions are met, it's suitable

            suitability_status = "Suitable" if suitable else "Not Suitable"

            return render_template('result.html', prediction=prediction_list, selection=modelSelected, suitability=suitability_status)

        except ValueError as e:
            return f"Error: Invalid input detected ({str(e)})", 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

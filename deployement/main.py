from flask import Flask,jsonify,request,render_template
import pandas as pd
import joblib
import numpy as np

model = joblib.load('./final_model.plk')

app =Flask(__name__, template_folder ='templates')
@app.route("/",methods=['GET'])
def home():
  return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
  try:
    data = request.get_json()
   # wine_data = data.get('wineData')
    fixed_acidity=data.get('fixed_acidity')
    volatile_acidity=data.get('volatile_acidity')
    citric_acid=data.get('citric_acid')
    residual_sugar=data.get('residual_sugar')
    chlorides=data.get('chlorides')
    free_sulfur_dioxide=data.get('free_sulfur_dioxide')
    total_sulfur_dioxide=data.get('total_sulfur_dioxide')
    density=data.get('density')
    pH=data.get('pH')
    sulphates=data.get('sulphates')
    alcohol=data.get('alcohol')
    
    #prediction stuff
    input = pd.DataFrame(data={
      'fixed acidity': [fixed_acidity],
      'volatile acidity': [volatile_acidity],
      'citric acid': [citric_acid],
      'residual sugar': [residual_sugar],
      'chlorides': [chlorides],
      'free sulfur dioxide': [free_sulfur_dioxide],
      'total sulfur dioxide': [total_sulfur_dioxide],
      'density': [density],
      'pH': [pH],
      'sulphates': [sulphates],
      'alcohol': [alcohol]
    })
    imput_data_as_array = np.asarray(input.values)

    reshaped_data = imput_data_as_array.reshape(1, -1)
    prediction = model.predict(reshaped_data)
    prediction_list = prediction.tolist()
    return jsonify({'result':prediction_list})
  
  
  
  except Exception as e:
    return jsonify({'error': str(e)})
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000)
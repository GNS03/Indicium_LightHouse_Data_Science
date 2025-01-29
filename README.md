### Exploratory Data Analysis

The exploratory data analysis is in `main.ipynb` and the final version of the model is in `model_pipeline.ipynb`.

# Requirements

To run the model locally, you will need:

- **Python 3.10+**: [Download here](https://www.python.org/downloads/)
- The libraries listed in the `requirements.txt` file:
    - If you already have Python installed, run the following in your terminal:
      ```sh
      pip install -r requirements.txt
      ```

# Downloading the Model

- Just clone this repository to your computer: [How to clone a GitHub repository](https://docs.github.com/pt/repositories/creating-and-managing-repositories/cloning-a-repository)
- If you only want the model, you can get it from this link: [Download Model](https://github.com/GNS03/Indicium_LightHouse_Data_Science/blob/master/model_pipeline.pkl)

# Making Predictions

You'll need a Python script in the same folder as the `.pkl` model file. Here's a template of code to make a prediction:

  ```python
  import pickle
  import pandas as pd
  
  with open("model_pipeline.pkl", "rb") as model_pkl:
      model = pickle.load(model_pkl)

  X_pred = pd.DataFrame([{
      'id': 2595,
      'nome': 'Skylit Midtown Castle',
      'host_id': 2845,
      'host_name': 'Jennifer',
      'bairro_group': 'Manhattan',
      'bairro': 'Midtown',
      'latitude': 40.75362,
      'longitude': -73.98377,
      'room_type': 'Entire home/apt',
      'minimo_noites': 1,
      'numero_de_reviews': 45,
      'ultima_review': '2019-05-21',
      'reviews_por_mes': 0.38,
      'calculado_host_listings_count': 2,
      'disponibilidade_365': 355
  }])
    
  if "id" in X_pred.columns:
      X_pred.drop(["id", "nome", "host_id", "host_name", "ultima_review"], axis=1, inplace=True)

  # Use the model to make a prediction
  predicted_price = model.predict(X_pred)
  print(f"Predicted price: {predicted_price[0].round(2)}")

# HousePricePrediction

This repository contains a simple machine-learning web application that predicts Boston house prices using a linear regression model trained on the Boston Housing dataset.

The project includes a Jupyter notebook used to explore the data and train the model, the serialized model and scaler for inference, and a small Flask web application that serves a form and a JSON API for predictions.

**Contents**
- **app.py**: Flask application that serves the web UI and the `/predict_api` endpoint.
- **templates/home.html**: Frontend form for entering feature values and displaying predictions.
- **HousePrice_Prediction.ipynb**: Notebook used for EDA, training, evaluation and pickling the model and scaler.
- **regmodel.pkl**: Pickled LinearRegression model used by the Flask app.
- **scaling.pkl**: Pickled StandardScaler used to scale inputs before prediction.
- **requirements.txt**: Python package dependencies.

**Features**
- Interactive web UI to enter the 13 Boston housing features and receive a predicted price.
- JSON API endpoint (`/predict_api`) for programmatic access.
- Reproducible training workflow in the notebook.

**Requirements**
- Python 3.8+ (this repo was developed with Python 3.14 in a virtual environment)
- pip

Quick install (recommended: use a virtual environment):

```bash
git clone https://github.com/<your-username>/HousePricePrediction.git
cd HousePricePrediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you already have a virtual environment active, just install the requirements:

```bash
pip install -r requirements.txt
```

**Run the app (development)**

1. Activate your virtual environment (if not already):

```bash
source venv/bin/activate
```

2. Start the Flask app:

```bash
python app.py
```

3. Open the UI in your browser at `http://127.0.0.1:5000`.

**API Usage**

Endpoint: `POST /predict_api`

Request body (JSON):

```json
{
	"data": {
		"crim": 0.00632,
		"zn": 18,
		"indus": 2.31,
		"chas": 0,
		"nox": 0.538,
		"rm": 6.575,
		"age": 65.2,
		"dis": 4.09,
		"rad": 1,
		"tax": 296,
		"ptratio": 15.3,
		"b": 396.9,
		"lstat": 4.98
	}
}
```

Important: the input features must be provided in the same order and with the same names as used during training: `['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat']`.

Example curl request:

```bash
curl -X POST http://127.0.0.1:5000/predict_api \
	-H "Content-Type: application/json" \
	-d '{"data": {"crim":0.00632, "zn":18, "indus":2.31, "chas":0, "nox":0.538, "rm":6.575, "age":65.2, "dis":4.09, "rad":1, "tax":296, "ptratio":15.3, "b":396.9, "lstat":4.98}}'
```

Response: JSON number (predicted median house value, units are same as training target — medv).

**Retraining / Notebook**

- Open `HousePrice_Prediction.ipynb` in Jupyter to explore the data and retrain the model. Execute the cells in order.
- After retraining, re-generate the model and scaler files by running the notebook cells that pickle `regmodel.pkl` and `scaling.pkl`.

If you prefer a script-based retrain process, create a script that follows the same steps as the notebook: load data, split, `fit` scaler on training data only, `transform` test data, train model, and pickle both `scaler` and `regression` objects.

**Troubleshooting**
- "ModuleNotFoundError: No module named 'flask'": activate the virtual environment and run `pip install -r requirements.txt`.
- "Port 5000 is in use": stop the process using the port, or run the app on a different port (edit `app.run(host='0.0.0.0', port=8080)` in `app.py` for example).
- Unexpected negative predictions or wildly incorrect values: ensure the scaler used for inference was fit only on training data. If you retrain the model, overwrite `regmodel.pkl` and `scaling.pkl` with the new pickles.

**Project structure**

```
.
├─ app.py
├─ HousePrice_Prediction.ipynb
├─ regmodel.pkl
├─ scaling.pkl
├─ requirements.txt
└─ templates/
	 └─ home.html
```

**License**

This project is provided under the MIT License — see the `LICENSE` file for details.

If you'd like, I can also add a short contribution guide, CI workflow, or a Dockerfile to containerize the app for deployment. Let me know which you'd prefer.



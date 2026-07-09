# Credit Risk Scoring Model for Banks

A full-stack machine learning application that predicts the probability of a loan default based on applicant details. 

**Live Demo:** [https://loan-prediction-app-delta.vercel.app](https://loan-prediction-app-delta.vercel.app)

## Features
- **Machine Learning Model:** Utilizes a Logistic Regression model trained on financial and demographic data to evaluate loan default risk.
- **Interactive UI:** A modern, clean frontend built with HTML, CSS, and Vanilla JavaScript that dynamically calculates the default probability in real-time.
- **RESTful API:** Powered by **FastAPI**, handling HTTP POST requests and returning JSON probabilities for seamless frontend integration.
- **Serverless Hosting:** Fully deployed on **Vercel** utilizing Python Serverless Functions.

## Tech Stack
- **Backend:** Python, FastAPI, Uvicorn, Pandas, Scikit-Learn, Joblib
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Vercel

## Project Structure
- `app.py`: The FastAPI application that exposes the `/api/predict` endpoint, scales incoming data, and runs it against the trained model.
- `train.py`: The machine learning training pipeline that preprocesses data, trains a `LogisticRegression` model, and saves the `.joblib` artifacts.
- `static/`: Contains the frontend assets (`index.html`, `style.css`, `script.js`).
- `vercel.json`: Configuration mapping for Vercel's Edge network to run the Python serverless functions correctly.

## Setup & Installation (Local Development)

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/Sarthak9Kastiya/credit-risk-scoring-model.git
cd credit-risk-scoring-model
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```
*The app will be accessible at `http://localhost:8000`.*

### 5. Training the model (Optional)
If you wish to train the model on your own dataset instead of using the provided mock data generator:
1. Place your dataset inside the project directory or modify the path in `train.py`.
2. Run `python train.py`.
3. The script will generate new `model.joblib`, `scaler.joblib`, and `features.joblib` files.

## How It Works
1. The user inputs their financial details via the frontend interface.
2. The frontend sends a POST request containing JSON data to the FastAPI `/api/predict` endpoint.
3. The backend applies the pre-fitted `StandardScaler` to the data to align with the training environment.
4. The trained `LogisticRegression` model calculates the probability of default and returns it.
5. The frontend dynamically updates the progress bar and displays the result to the user.

## License
MIT License

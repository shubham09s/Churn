from flask import Flask, render_template, request
from model import predict_churn

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	data = [
		int(request.form['Age']),
		int(request.form['Tenure']),
		float(request.form['MonthlyCharges']),
		float(request.form['TotalCharges']),
		int(request.form['SupportCalls'])
	]

	prediction = predict_churn(data)

	if prediction == 1:
		result = "Customer Will Churn ❌"
	else:
		result = "Customer Will Stay ✅"

	return render_template('index.html', prediction=result)

if __name__ == "__main__":
	app.run(debug=True)

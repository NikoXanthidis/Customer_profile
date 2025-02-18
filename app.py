from flask import Flask, render_template, request, jsonify
import torch
import joblib
import numpy as np

#Vamos definir o modelo (NeuralNet) / Let's define the model (NeuralNet)
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Vamos carregar scaler e modelo criados no codigo anterior / Let's load scaler and model created in the previous code.
scaler = joblib.load("scaler.pkl")
modelo = NeuralNet(input_size=17, num_classes=10)  # Ajuste conforme necessário
modelo.load_state_dict(torch.load("modelo_credito.pth"))
modelo.eval()

# Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prever", methods=["POST"])
def prever():
    try:
        dados = request.get_json()

        # Converter os dados para um array numpy / Convert the data to a numpy array.
        features = np.array([[
            dados["BALANCE"], dados["BALANCE_FREQUENCY"], dados["PURCHASES"],
            dados["ONEOFF_PURCHASES"], dados["INSTALLMENTS_PURCHASES"], dados["CASH_ADVANCE"],
            dados["PURCHASES_FREQUENCY"], dados["ONEOFF_PURCHASES_FREQUENCY"],
            dados["PURCHASES_INSTALLMENTS_FREQUENCY"], dados["CASH_ADVANCE_FREQUENCY"],
            dados["CASH_ADVANCE_TRX"], dados["PURCHASES_TRX"], dados["CREDIT_LIMIT"],
            dados["PAYMENTS"], dados["MINIMUM_PAYMENTS"], dados["PRC_FULL_PAYMENT"],
            dados["TENURE"]
        ]])

        # Normalizar os dados / Normalize the data.
        features_scaled = scaler.transform(features)

        # Converter para tensor do PyTorch / Convert to PyTorch tensor.
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        # Fazer a previsão / Make the prediction.
        with torch.no_grad():
            outputs = modelo(features_tensor)
            _, predicted = torch.max(outputs, 1)

        # Retornar a resposta como JSON / Return the response as JSON.
        return jsonify({"cluster": int(predicted.item())})

    except Exception as e:
        return jsonify({"erro": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

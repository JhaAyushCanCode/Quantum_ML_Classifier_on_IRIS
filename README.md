# Quantum-Classical Hybrid Model on Iris Dataset
This project demonstrates a simple Quantum Machine Learning (QML) model that combines classical preprocessing with a quantum neural network using PennyLane. It classifies samples from the Iris dataset using just two input features and a small quantum circuit.

# Project Highlights
Hybrid model using PennyLane, NumPy, and scikit-learn

Applies AngleEmbedding and StronglyEntanglingLayers for quantum processing

Classical softmax output layer

Trained using Adam Optimizer

Visualizes both quantum circuit and model accuracy over training epochs

📁 Files
QML_QML_liteData.py: Main script containing data processing, quantum model, training, and evaluation logic.

🧠 Model Workflow
Load Iris Dataset
Uses only 2 features for simplicity.

Preprocessing

Scales features using StandardScaler

One-hot encodes class labels (3 classes)

Quantum Circuit

2-qubit circuit built using:

AngleEmbedding for input

StronglyEntanglingLayers for trainable parameters

Outputs expectation values of Pauli-Z operators

Softmax + Output Layer

Classical linear layer maps quantum outputs to class probabilities

Applies softmax for multiclass classification

Training

Mini-batch gradient descent using qml.AdamOptimizer

Tracks accuracy every 5 epochs

Evaluation

Final accuracy on test set

Accuracy plot over training

📊 Visuals
Quantum Circuit diagram via qml.draw_mpl

Accuracy Curve plotted using matplotlib

🧪 Installation
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/quantum-iris-classifier.git
cd quantum-iris-classifier
2. Create & activate virtual environment (optional but recommended)
bash
Copy
Edit
python -m venv qenv
source qenv/bin/activate   # or use `qenv\Scripts\activate` on Windows
3. Install dependencies
bash
Copy
Edit
pip install pennylane matplotlib scikit-learn
▶️ Run the Script
bash
Copy
Edit
python QML_QML_liteData.py
📝 Example Output
yaml
Copy
Edit
Epoch 0 : Test Accuracy = 0.72
Epoch 5 : Test Accuracy = 0.80
...
Final Test Accuracy : 0.90
📌 Requirements
Python ≥ 3.7

PennyLane

NumPy

scikit-learn

matplotlib

🧠 Notes
Uses only 2 features from Iris for visualization and circuit simplicity.

Can be extended to 4 features using more qubits.

Not optimized for real quantum hardware (uses default.qubit simulator).

📬 Contact
For questions or suggestions, feel free to reach out or open an issue!

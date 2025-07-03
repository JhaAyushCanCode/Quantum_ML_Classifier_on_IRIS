import pennylane as qml 
from pennylane import numpy as np
import numpy as onp                                                            #Ordinary Numpy
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt






# Loading data and selecting 2 features for simplicity

data = load_iris() 
X = data.data[:, :2] 
Y = data.target.reshape(-1, 1) 

# Standardize input features 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

# OneHot Encode the labels : Example - [1,0,0] for class 0

encoder = OneHotEncoder(sparse_output= False) 
Y_encoded = encoder.fit_transform(Y) 

#Split into training and testing sets 

X_train , X_test , Y_train , Y_test = train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42) 






# Setting Up Quantum Circuit 

n_qubits = 2
dev = qml.device("default.qubit", wires= n_qubits)

# Building the Quantum Circuit 
#   *AngleEmbedding - to i/p data into qubits 
#   *StronglyEntanglingLayers - to entangle and rotate qubits 

@qml.qnode(dev) 

def quantum_circuit (inputs, weights) :
    qml.templates.AngleEmbedding(inputs, wires= range(n_qubits)) 
    qml.templates.StronglyEntanglingLayers(weights, wires = range(n_qubits))
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

#Visualize Quantum circuit with example inputs and weights
n_layers = 2
example_inputs = np.array([0.1,0.2] , requires_grad = False) 
example_weights = np.random.randn (n_layers, n_qubits, 3)

qml.drawer.use_style("black_white") 

fig, ax = qml.draw_mpl(quantum_circuit)(example_inputs, example_weights)

plt.show()

#Define Prediction Logic 
# Add softmax and prediction function 

def softmax(x) : 
    e_x = np.exp(x - np.max(x))
    
    return e_x / np.sum (e_x, axis = 1 , keepdims = True) 


def predict(X, weights) :
    preds = [quantum_circuit(x, weights) for x in X] 
    
    logits = np.array(preds) 
    
    probs = softmax(logits @ W_output + b_output) 
    
    return probs

#Initialize Parameters 

n_layers = 2
n_classes = 3

np.random.seed(42)

weights = np.random.randn(n_layers, n_qubits, 3, requires_grad = True) 

W_output = np.random.randn(n_qubits, n_classes) 

b_output = np.random.randn(n_classes)





# Training the Model 

opt = qml.AdamOptimizer(stepsize=0.05) 

epochs = 50
batch_size = 16
accuracy_history = [] 

for epoch in range(epochs) :
    
    batch_index = onp.random.randint(0, len(X_train), batch_size) 
    X_batch = X_train [batch_index] 
    Y_batch = Y_train [batch_index] 
    
    def cost(weights) :
        preds = predict(X_batch, weights) 
        
        return -np.mean(np.sum(Y_batch*np.log(preds + 1e-10) , axis = 1)) 
    
    weights = opt.step(cost, weights) 
    
    if epoch % 5 == 0 :
        Y_pred = predict(X_train, weights) 
        acc = accuracy_score(onp.argmax(Y_train, axis=1), onp.argmax(Y_pred, axis=1)) 
        accuracy_history.append(acc) 
        
        print (f"Epoch {epoch} : Test Accuracy = {acc:.2f}") 
       
        
       
       

#Evaluating Final Model 

Y_pred_final = predict(X_test, weights)
final_acc = accuracy_score(onp.argmax(Y_test, axis=1), onp.argmax(Y_pred_final, axis=1))

print (f"Final Test Accuracy : {final_acc:.2f}")

#Plot Accuracy Over Epochs 

plt.plot(range(0, epochs, 5), accuracy_history, marker = 'o')

plt.xlabel("Epoch")
plt.ylabel ("Test Accuracy")

plt.title("Quantum-Classical Model Accuracy Over Epochs")

plt.grid(True)
plt.show() 
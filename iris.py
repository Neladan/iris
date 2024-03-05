import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from mlxtend.plotting import plot_decision_regions


# Charger le fichier CSV dans un DataFrame
df = pd.read_csv('iris_dataset.csv')

def charger(nom):
    X = []
    Y = []
    T = open("iris_dataset.csv", "r").readlines()
    for i in T[1:]:
        a = i.replace("\n","").split(",")
        X.append([float(a[0]), float(a[1])])
        if nom == a[2]:
            Y.append(1)
        else:
            Y.append(0)
    return np.array(X), np.array(Y)

# Diviser les données en fonction de l'espèce
setosa_data = df[df['species'] == 'Iris-setosa']
versicolor_data = df[df['species'] == 'Iris-versicolor']
virginica_data = df[df['species'] == 'Iris-virginica']


# Fonction pour entraîner un classificateur binaire
def train_binary_classifier(target_species):
    # Diviser les données en fonction des caractéristiques et de la cible
    X , y = charger(target_species)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # print(y_train)

    # Standardiser les données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Créer et entraîner le modèle LinearSVC

    md = LinearSVC(C=1.0, dual=False)

    md.fit(X_train, y_train)


    # Faire des prédictions sur l'ensemble de test
    pred = md.predict(X_test)

    # Calculer la précision du modèle et le recall score
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    # Display it
    print(target_species + " Precision: ", precision)
    print(target_species + " Recall: ", recall)

    #matrix confusion, calculate and display
    cm = confusion_matrix(y_test, pred)

    # format d'annotation avec fmt=d, x,y ticklabels pour l'étiquettage
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not ' + target_species, target_species],
                yticklabels=['Not ' + target_species, target_species])

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {target_species}')
    plt.show()

    # Calulate and display le ROC Curve
    return md



# Frontière de décision
def plot_decision_boundary(model, target_species):

    # model = train_binary_classifier(target_species)

    X , y = charger(target_species)
    y = y.astype(int)

    # Display classifier model
    plot_decision_regions(X, y, clf=model, legend=2)

    plt.xlabel('Longueur du pétale (cm)')
    plt.ylabel('Largeur du pétale (cm)')
    plt.title(f'Frontière de décision pour {target_species}')
    plt.show()


# Entraîner les classificateurs pour chaque espèce
setosa_classifier = train_binary_classifier('Iris-setosa')
versicolor_classifier = train_binary_classifier('Iris-versicolor')
virginica_classifier = train_binary_classifier('Iris-virginica')

# Display plot boundary
plot_decision_boundary(setosa_classifier, 'Iris-setosa')
plot_decision_boundary(versicolor_classifier, 'Iris-versicolor')
plot_decision_boundary(virginica_classifier, 'Iris-virginica')


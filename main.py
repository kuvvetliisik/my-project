import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from tkinter import *
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

df = pd.read_csv('C:/Users/sakir/PycharmProjects/YapayZeka/breast_cancer/breast-cancer.data')

df_encoded = pd.get_dummies(df, columns=df.columns, drop_first=True)

y = df_encoded['no-recurrence-events_recurrence-events']
X = df_encoded.drop(columns=['no-recurrence-events_recurrence-events'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

loss, accuracy = model.evaluate(X_test, y_test)
print('Yapay Sinir Ağı Doğruluğu:', accuracy)

def plot_graphs():
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    axes[0].plot(history.history['loss'], label='Eğitim Kaybı')
    axes[0].plot(history.history['val_loss'], label='Doğrulama Kaybı')
    axes[0].set_title('Model Kaybı')
    axes[0].set_xlabel('Dönemler')
    axes[0].set_ylabel('Kayıp')
    axes[0].legend()

    axes[1].plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    axes[1].plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    axes[1].set_title('Model Doğruluğu')
    axes[1].set_xlabel('Dönemler')
    axes[1].set_ylabel('Doğruluk')
    axes[1].legend()

    y_pred = (model.predict(X_test) >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[2], cmap='Blues')
    axes[2].set_title('Karışıklık Matrisi')
    axes[2].set_xlabel('Tahmin Edilen')
    axes[2].set_ylabel('Gerçek')

    age_distribution = df['30-39'].value_counts()
    axes[3].pie(age_distribution, labels=age_distribution.index, autopct='%1.1f%%', startangle=140)
    axes[3].set_title('Yaş Dağılımı')

    return fig

def predict_cancer():
    try:
        age = age_var.get()
        menopause = menopause_var.get()
        tumor_size = tumor_size_var.get()
        inv_nodes = inv_nodes_var.get()
        node_caps = node_caps_var.get()
        deg_malig = deg_malig_var.get()
        breast = breast_var.get()
        breast_quad = breast_quad_var.get()
        irradiat = irradiat_var.get()

        input_data = [
            1 if age == "30-39" else 0,
            1 if age == "40-49" else 0,
            1 if age == "50-59" else 0,
            1 if age == "60-69" else 0,
            1 if age == "70-79" else 0,
            1 if menopause == "lt40" else 0,
            1 if menopause == "ge40" else 0,
            1 if tumor_size == "0-4" else 0,
            1 if tumor_size == "5-9" else 0,
            1 if tumor_size == "10-14" else 0,
            1 if tumor_size == "15-19" else 0,
            1 if tumor_size == "20-24" else 0,
            1 if tumor_size == "25-29" else 0,
            1 if tumor_size == "30-34" else 0,
            1 if tumor_size == "35-39" else 0,
            1 if tumor_size == "40-44" else 0,
            1 if tumor_size == "45-49" else 0,
            1 if tumor_size == "50-54" else 0,
            1 if inv_nodes == "0-2" else 0,
            1 if inv_nodes == "3-5" else 0,
            1 if inv_nodes == "6-8" else 0,
            1 if inv_nodes == "9-11" else 0,
            1 if inv_nodes == "12-14" else 0,
            1 if node_caps == "yes" else 0,
            1 if node_caps == "no" else 0,
            1 if deg_malig == "1" else 0,
            1 if deg_malig == "2" else 0,
            1 if deg_malig == "3" else 0,
            1 if breast == "left" else 0,
            1 if breast == "right" else 0,
            1 if breast_quad == "left_up" else 0,
            1 if breast_quad == "left_low" else 0,
            1 if breast_quad == "right_up" else 0,
            1 if breast_quad == "right_low" else 0,
            1 if irradiat == "yes" else 0,
            1 if irradiat == "no" else 0,
        ]

        input_data = input_data[:34]

        input_data = np.array(input_data).reshape(1, -1)

        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        prediction = prediction[0][0]
        result = "Kötü Huylu" if prediction >= 0.5 else "İyi Huylu"

        if result == "Kötü Huylu":
            root.configure(bg='red')
            messagebox.showinfo("Tahmin", f"Kansere ait tahmin sonucu: {result}")
        else:
            root.configure(bg='green')
            messagebox.showinfo("Tahmin", f"Kansere ait tahmin sonucu: {result}")
    except Exception as e:
        messagebox.showerror("Hata", str(e))

root = Tk()
root.title("Meme Kanseri Tahmini")

Label(root, text="Aşağıdaki özellikler için değerleri girin:").grid(row=0, columnspan=2)

age_var = StringVar(root)
age_var.set("30-39")
menopause_var = StringVar(root)
menopause_var.set("lt40")
tumor_size_var = StringVar(root)
tumor_size_var.set("0-4")
inv_nodes_var = StringVar(root)
inv_nodes_var.set("0-2")
node_caps_var = StringVar(root)
node_caps_var.set("Evet")
deg_malig_var = StringVar(root)
deg_malig_var.set("1")
breast_var = StringVar(root)
breast_var.set("Sol")
breast_quad_var = StringVar(root)
breast_quad_var.set("Sol_Üst")
irradiat_var = StringVar(root)
irradiat_var.set("Evet")

Label(root, text="Yaş").grid(row=1, column=0)
OptionMenu(root, age_var, "30-39", "40-49", "50-59", "60-69", "70-79").grid(row=1, column=1)
Label(root, text="Menopoz").grid(row=2, column=0)
OptionMenu(root, menopause_var, "lt40", "ge40").grid(row=2, column=1)
Label(root, text="Tümör Boyutu").grid(row=3, column=0)
OptionMenu(root, tumor_size_var, "0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54").grid(row=3, column=1)
Label(root, text="Lenf Nodları").grid(row=4, column=0)
OptionMenu(root, inv_nodes_var, "0-2", "3-5", "6-8", "9-11", "12-14").grid(row=4, column=1)
Label(root, text="Node Caps").grid(row=5, column=0)
OptionMenu(root, node_caps_var, "Evet", "Hayır").grid(row=5, column=1)
Label(root, text="Derece").grid(row=6, column=0)
OptionMenu(root, deg_malig_var, "1", "2", "3").grid(row=6, column=1)
Label(root, text="Göğüs").grid(row=7, column=0)
OptionMenu(root, breast_var, "Sol", "Sağ").grid(row=7, column=1)
Label(root, text="Göğüs Kısmı").grid(row=8, column=0)
OptionMenu(root, breast_quad_var, "Sol_Üst", "Sol_Alt", "Sağ_Üst", "Sağ_Alt").grid(row=8, column=1)
Label(root, text="Işın Tedavisi").grid(row=9, column=0)
OptionMenu(root, irradiat_var, "Evet", "Hayır").grid(row=9, column=1)

Button(root, text="Tahmin Et", command=predict_cancer).grid(row=10, columnspan=2)
Button(root, text="Grafikleri Göster", command=lambda: show_graphs()).grid(row=11, columnspan=2)

def show_graphs():
    fig = plot_graphs()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=12, columnspan=2)
    canvas.draw()

root.mainloop()

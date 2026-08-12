import tkinter as tk
from tkinter import messagebox
import pandas as pd
import pickle

# Load the trained model
try:
    with open("house_price_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    messagebox.showerror("Error", "house_price_model.pkl not found!")
    exit()

# Prediction Function
def predict_price():
    try:
        overall = float(overall_quality.get())
        area = float(gr_liv_area.get())
        garage = float(garage_cars.get())

        # Create DataFrame with same column names used in training
        input_data = pd.DataFrame({
            "Overall Qual": [overall],
            "Gr Liv Area": [area],
            "Garage Cars": [garage]
        })

        prediction = model.predict(input_data)

        result.config(
            text=f"Predicted House Price\n₹ {prediction[0]:,.2f}",
            fg="green"
        )

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values!")

# Main Window
root = tk.Tk()
root.title("House Price Prediction")
root.geometry("500x450")
root.resizable(False, False)

# Heading
tk.Label(
    root,
    text="House Price Prediction",
    font=("Arial", 18, "bold")
).pack(pady=15)

# Overall Quality
tk.Label(root, text="Overall Quality").pack()
overall_quality = tk.Entry(root, width=30)
overall_quality.pack(pady=5)

# Ground Living Area
tk.Label(root, text="Ground Living Area (sq ft)").pack()
gr_liv_area = tk.Entry(root, width=30)
gr_liv_area.pack(pady=5)

# Garage Cars
tk.Label(root, text="Garage Cars").pack()
garage_cars = tk.Entry(root, width=30)
garage_cars.pack(pady=5)

# Predict Button
tk.Button(
    root,
    text="Predict Price",
    command=predict_price,
    bg="green",
    fg="white",
    font=("Arial", 12, "bold")
).pack(pady=20)

# Result
result = tk.Label(
    root,
    text="Predicted Price will appear here",
    font=("Arial", 13)
)
result.pack()

# Run
root.mainloop()
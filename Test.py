import time
import joblib
import os
from datetime import datetime
import re

start_time = datetime.now()
print("Started Time :",start_time)

class Test_on_model:

    def Test_on_data(self):
        try:
            model = joblib.load("Trained_model/Trained_model.pkl")

            while True:
                user_input = input("Write your feelings here :")
                if user_input.strip().lower() == "exit":
                    break

                user_input = re.sub(r"[^\w\s]","",user_input)

                """label_map_en = {
                    0: "happy",
                    1: "sad",
                    2: "hate",
                    3: "ungrateful",
                    4: "romantic",
                    5: "fear",
                    6: "trust",
                    7: "distrust",
                    8: "expectation",
                    9: "excited",
                    10: "indifference",
                    11: "tired",
                    12: "unknown"  
                }"""

                prediction = model.predict([user_input])[0]
                print("Prediction :", prediction)

        except Exception as e:
            print(f"Model Failed to Predict: {e}")

C = Test_on_model()
C.Test_on_data()









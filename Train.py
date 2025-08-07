from dataset import dataset_Analyse
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline , make_pipeline
import seaborn as sns
from datetime import datetime
import logging
import matplotlib
import joblib
from matplotlib import font_manager
import pandas as pd


matplotlib.use("TkAgg")
font_path = 'fonts/Vazir-Regular-FD.ttf'
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

start_time = datetime.now()
print("Start time:", start_time)

logging.basicConfig(
    filename='Logs/Train.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

analyzer = dataset_Analyse()
analyzer.Dataset()
cleaned_data = analyzer.Cleaned_DS()


class Main_model:

    def Train(self):
        global cleaned_data
        try:
            x = cleaned_data["sentence"]
            y = cleaned_data["emotion"]

            df = pd.DataFrame({'sentence': x, 'emotion': y})
            df = df.dropna(subset=['sentence', 'emotion'])

            x = df['sentence']
            y = df['emotion']

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
            model_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=2500, stop_words='english')),
                ('clf', LogisticRegression(max_iter=3000, solver='saga'))
            ])


            model_pipeline.fit(x_train,y_train)

            labels = model_pipeline.classes_ if hasattr(model_pipeline.named_steps['clf'], 'classes_') else sorted(
                y.unique())

            y_pred = model_pipeline.predict(x_test)

            print("Accuracy Score :", accuracy_score(y_test, y_pred))
            print("Classification Report :", classification_report(y_test, y_pred))

            cmap = sns.color_palette("tab20c", as_cmap=True)

            confusion = confusion_matrix(y_test, y_pred, labels=labels)
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion, annot=True, fmt='d', cmap=cmap,
                        xticklabels=labels,
                        yticklabels=labels,
                        linewidths=0.5,
                        linecolor='gray',
                        cbar=True)

            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig("Confusion_matrix.png")
            time.sleep(3)
            plt.close()
            print("Successfully Train Model ..")

            try:
                joblib.dump(model_pipeline, "Trained_model/Trained_model.pkl")
                print(f"Succesfully Saved model :{model_pipeline}")
            except Exception as e:
                print(f"Cant Save model due to : {e}")
                return False

        except Exception as e:
            print(f"Failed to Train Model due to : {e}")
            return False


C = Main_model()
C.Train()

end_time = datetime.now()
print("End time:", end_time)


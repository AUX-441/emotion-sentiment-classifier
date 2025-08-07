import pandas as pd # importing Pandas module


class dataset_Analyse:

    def __init__(self):
        self.data = None # set data as None
        self.cleaned = None # set Final Cleaned Data as None

    def Dataset(self):
        try:
            self.data = pd.read_csv("Dataset/sentiment.csv") # read main_Dataset = 1 M for test

            print("Successfully Read Dataset") # loaded Dataset
            print("Columns:", self.data.columns.tolist())  # show names of DataFrame
            print("Emotions:", self.data['emotion'].unique()) # show all emotion Labels
            print("Emotion Count:", self.data['emotion'].nunique()) # show Length of the emotion Labels
            print("Emotion Value Counts:\n", self.data['emotion'].value_counts()) # show counts of emotion per Labels
            print("Unique Sentences:", self.data['sentence'].nunique()) # show number of unique sentence

            dup = self.data['sentence'].value_counts() # show count of sentence Label
            print("Repeated Sentences (>3 times):\n", dup[dup > 3]) # Show every text have repeated 3 times

            return self.data # return data

        except Exception as e:
            print(f"Error while loading dataset: {e}")
            return None # Return None except

    def Cleaned_DS(self):
        if self.data is None:
            print("Data not loaded. Loading now...")
            self.Dataset() # Load dataset from Upper side Function which returned

        try:
            print("Cleaning Dataset...") # starting to clean data
            value_cnts = self.data["sentence"].value_counts() # get all values of sentence Label
            repeated = value_cnts[value_cnts > 3].index # search for textx which repeated 3 times
            self.cleaned = self.data[~self.data["sentence"].isin(repeated)] # Delete textx with repeated more than 3 times
            print("Cleaning Done. New Shape:", self.cleaned.shape) # show Success Cleaned Data Print
            return self.cleaned # return Final Cleaned Data

        except Exception as e:
            print(f"Failed to Clean Dataset: {e}")
            return None # Return None except

# ** usage on other module **

# from dataset import dataset_Analyse

# analyzer = dataset_Analyse()
# main_data = analyzer.Dataset()
# cleaned = analyzer.Cleaned_DS()
# print(cleaned.head())





import matplotlib.pyplot as plt
import metrics
import warnings
warnings.filterwarnings("ignore")

data, target = metrics.preprocess()

barHeights = metrics.UnivariateSelection(data, target)
features = ['age', 'blood pressure',  'specific gravity', 'albumin', 'sugar', 'red blood cells', 'pus cell',
            'pus cell clumps', 'bacteria', 'blood glucose random', 'blood urea', 'serum creatinine', 'sodium',
            'potassium', 'hemoglobin', 'packed cell volume', 'white blood cell count', 'red blood cell count',
            'hypertension', 'diabetes mellitus', 'coronary artery disease', 'appetite', 'pedal edema', 'anemia']
fig, ax = plt.subplots()
plt.bar(range(1, 25), barHeights)
plt.subplots_adjust(bottom=0.28, left=0.1)
plt.yticks([0, 0.05, 0.1, 0.15, 0.20], ["0%", "5%", "10%", "15%", "20%"], size=7)
plt.xticks(range(1, 25), features, rotation=270, size=7)
plt.title('Importance of Each Feature', size=16)
plt.ylabel('Importance (%)', size=8)
plt.show()

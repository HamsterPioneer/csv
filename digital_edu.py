import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('train.csv')
df.info()
df.drop(['followers_count','life_main','has_photo','has_mobile','people_main','last_seen'], axis = 1, inplace = True)
df['langs'].fillna('Русский', inplace = True)
print(df['langs'].value_counts())

def graduate(gd):
    if gd > 2030 or gd < 1900:
        return 2000
    else:
        return gd

df['graduation'] = df['graduation'].apply(graduate)

def lang(lg):
    if lg == "False" or "Русский" in lg:
        return 1
    else:
        return 0 

df['langs'] = df['langs'].apply(lang)

x = df.drop('result', axis = 1)
y = df['result']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

scam = StandardScaler()
x_train = scam.fit_transform(x_train)
x_test = scam.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, x_test)

y_pred = classifier.predict(x_test)

dd = accuracy_score(y, y_pred)

print(dd)


import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

train_data = json.load(open('data/train.json'))
"""
  {
    "id": 10259,
    "cuisine": "greek",
    "ingredients": [
      "romaine lettuce",
      "black olives",
      "grape tomatoes",
      "garlic",
      "pepper",
      "purple onion",
      "seasoning",
      "garbanzo beans",
      "feta cheese crumbles"
    ]
  },
"""
ingreds = list({i for r in train_data for i in r['ingredients']})

print(f'have {len(ingreds):,} ingredients')

tr = []

for r in train_data:
	main_ingrs = {i: 1 for i in r['ingredients']}
	more_ingrs = {w: 1 for i in main_ingrs for w in i.split() if w.isalpha()}
	tr.append({**{'id': r['id'], 'cuisine': r['cuisine']}, 
					**main_ingrs, **more_ingrs})

le = LabelEncoder()

train_data_df = pd.DataFrame.from_dict(tr).fillna(0)

le.fit(train_data_df['cuisine'])

train_data_df['cuisine'] = le.transform(train_data_df['cuisine'])

X_train, X_test, y_train, y_test = train_test_split(train_data_df.drop('cuisine', axis=1), train_data_df['cuisine'], 
          test_size=0.3, random_state=42)

print(X_train.shape)

clf = RandomForestClassifier(max_depth=2, random_state=5)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))


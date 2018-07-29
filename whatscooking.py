import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

class Coookings:

  def __init__(self):

    self.train_data = json.load(open('data/train.json'))
    self.test_data = json.load(open('data/test.json'))

    cuis_ = [_['cuisine'] for _ in self.train_data]
    ids_ = {_['id'] for _ in self.train_data}
    ingrs_ = {i.lower().strip() for _ in self.train_data for i in _['ingredients']}

    print(f'records: {len(self.train_data):,}\ncuisines: {len(set(cuis_))}\ningredients: {len(ingrs_):,}')
    print(sorted([(k, v) for k, v in Counter(cuis_).items()], key=lambda _: _[1], reverse=True))

  def make_matrix(self, )

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

if __name__ == '__main__':

  cook = Coookings()
# tr = []

# for r in train_data:
# 	main_ingrs = {i: 1 for i in r['ingredients']}
# 	more_ingrs = {w: 1 for i in main_ingrs for w in i.split() if w.isalpha()}
# 	tr.append({**{'id': r['id'], 'cuisine': r['cuisine']}, 
# 					**main_ingrs, **more_ingrs})

# le = LabelEncoder()

# train_data_df = pd.DataFrame.from_dict(tr).fillna(0)

# le.fit(train_data_df['cuisine'])

# train_data_df['cuisine'] = le.transform(train_data_df['cuisine'])

# X_train, X_test, y_train, y_test = train_test_split(train_data_df.drop('cuisine', axis=1), train_data_df['cuisine'], 
#           test_size=0.3, random_state=42, stratify=train_data_df['cuisine'])

# print(X_train.shape)

# clf = RandomForestClassifier(max_depth=2, random_state=5)

# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)

# print(accuracy_score(y_test, y_pred))


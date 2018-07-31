import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline, make_union

from sklearn.base import BaseEstimator, TransformerMixin

class C1(BaseEstimator, TransformerMixin):

		def fit(self, X, y=None, **fit_params):
				return self

		def transform(self, X, **transform_params):
				return [_ for _ in X['ingredients'].str.split('.')]


class Coookings:

	def __init__(self):

		self.train_data = json.load(open('data/train.json'))
		self.test_data = json.load(open('data/test.json'))

		# note that test data looks similar to train data but there are NO cuisine fields

		cuis_ = [_['cuisine'] for _ in self.train_data]
		ids_ = {_['id'] for _ in self.train_data}
		ingrs_ = {i.lower().strip() for _ in self.train_data for i in _['ingredients']}

		print(f'records: {len(self.train_data):,}\ncuisines: {len(set(cuis_))}\ningredients: {len(ingrs_):,}')
		print(sorted([(k, v) for k, v in Counter(cuis_).items()], key=lambda _: _[1], reverse=True))

		self.train_ = pd.DataFrame.from_dict([{'id': r['id'], 
																						'ingredients': '. '.join(r['ingredients']), 'cuisine': r['cuisine']}  
							for r in self.train_data])

	def split(self):

		X = self.train_['ingredients'].str.split('.')
		y = self.train_['cuisine']

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, 
					test_size=0.3, random_state=42, stratify=y)

		print(self.X_train.head())

		return self

	def train_model(self):

		# cv = CountVectorizer(strip_accents='ascii', analyzer='word', ngram_range=(1,2))

		# cv.fit(self.X_train['ingredients'].str.split('.'))

		# print(cv.vocabulary_)


		pipeline = make_pipeline(C1, CountVectorizer(strip_accents='ascii', analyzer='word', ngram_range=(1,2)),
															RandomForestClassifier())
		# print(len(self.X_train.shape))
		# print(self.y_train.shape)

		# d = self.X_train['ingredients'].tolist()

		# print(d[:20])

		# print(self.y_train[:20])

		pipeline.fit(self.X_train)
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

	cook = Coookings().split().train_model()

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



# print(X_train.shape)

# clf = RandomForestClassifier(max_depth=2, random_state=5)

# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)

# print(accuracy_score(y_test, y_pred))


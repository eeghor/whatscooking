import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline, make_union
from itertools import chain

from sklearn.base import BaseEstimator, TransformerMixin

class C1(BaseEstimator, TransformerMixin):

		def fit(self, X, y=None, **fit_params):
				return self

		def transform(self, X, **transform_params):
				lst = [_.strip() for _ in chain.from_iterable(X.str.split('.').tolist())]
				print(lst)
				return lst


class Coookings:

	def __init__(self):

		self.train_data = pd.read_json('data/train.json', orient='records')
		self.test_data = pd.read_json('data/test.json', orient='records')

		# note that test data looks similar to train data but there are NO cuisine fields

	def split(self):

		X = self.train_data['ingredients'].str.join('. ')
		y = self.train_data['cuisine']

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, 
					test_size=0.3, random_state=42, stratify=y)

		print(f'testing set: {self.X_train.shape}')


		return self

	def train_model(self):

		pipeline = make_pipeline(CountVectorizer(strip_accents='ascii', analyzer='word', ngram_range=(1,2)),
															RandomForestClassifier())

		print(self.X_train.head())
		print(self.X_train.shape)
		print(len(self.X_train.shape))
		pipeline.fit(self.X_train)

if __name__ == '__main__':

	cook = Coookings().split().train_model()
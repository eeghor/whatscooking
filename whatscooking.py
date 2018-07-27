import json
import pandas as pd

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
ingreds = list({i for r in train_data for i in r})

print(f'have {len(ingreds)} ingredients')
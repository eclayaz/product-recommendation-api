# Databricks notebook source
# MAGIC %md
# MAGIC # Product Recommendation Notebook
# MAGIC
# MAGIC This notebook helps:
# MAGIC 1. Load product data with attributes.
# MAGIC 2. Select 3 diverse initial products.
# MAGIC 3. Allow user feedback to update attribute weights.
# MAGIC 4. Recommend products based on updated weights over iterations.
# MAGIC 5. After 4 iterations, generate a prompt describing desired product features.

# COMMAND ----------


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import random
from typing import List, Dict, Optional
from pydantic import BaseModel


# COMMAND ----------


# Load your product data. Ensure the CSV has columns: 'product_id' and attribute columns.
df = pd.read_csv('products.csv')
attributes = [col for col in df.columns if col != 'product_id']
X = df[attributes].values
ids = df['product_id'].values


# COMMAND ----------


# Select 3 diverse products using farthest point sampling
selected = [random.randrange(len(ids))]
for _ in range(2):
    dist = cosine_distances(X, X[selected]).min(axis=1)
    next_idx = dist.argmax()
    selected.append(next_idx)
initial_ids = ids[selected]
print("Initial diverse product IDs:", initial_ids)
selected_indices = selected  # Keep track of current selection indices


# COMMAND ----------


# Initialize attribute weights
weights = np.zeros(len(attributes))

# Run up to 4 iterations of feedback
for iteration in range(1, 5):
    print(f"\nIteration {iteration}")

    # Display current products
    display(df.iloc[selected_indices][['product_id'] + attributes])

    # Get user feedback: liked product IDs
    liked_input = input("Enter liked product IDs separated by commas: ")
    liked_ids = [lid.strip() for lid in liked_input.split(",")]

    # Update weights: +1 for attributes in liked products, -1 for attributes in others
    selected_ids = ids[selected_indices]
    selected_X = X[selected_indices]
    liked_mask = np.isin(selected_ids, liked_ids)
    weights += selected_X[liked_mask].sum(axis=0)
    weights -= selected_X[~liked_mask].sum(axis=0)

    # Score all products
    scores = X.dot(weights) + np.random.randn(len(ids)) * 0.01
    # Select top 3 products
    selected_indices = np.argsort(scores)[-3:][::-1]
    print("Next recommended product IDs:", ids[selected_indices])

# After 4 iterations, generate a prompt based on positive-weight attributes
positive_attributes = [attributes[i] for i, w in enumerate(weights) if w > 0]
prompt = "Looking for products with features: " + ", ".join(positive_attributes)
print("\nGenerated prompt:")
print(prompt)

class ProductRecommendation:
    def __init__(self, csv_path: str = 'products.csv'):
        # Load product data
        self.df = pd.read_csv(csv_path)
        self.attributes = [col for col in self.df.columns if col != 'product_id']
        self.X = self.df[self.attributes].values
        self.ids = self.df['product_id'].values
        self.weights = np.zeros(len(self.attributes))
        self.selected_indices = []
        self.iteration = 0
        self.initialize_recommendations()

    def initialize_recommendations(self) -> List[str]:
        """Initialize with 3 diverse products"""
        self.selected_indices = [random.randrange(len(self.ids))]
        for _ in range(2):
            dist = cosine_distances(self.X, self.X[self.selected_indices]).min(axis=1)
            next_idx = dist.argmax()
            self.selected_indices.append(next_idx)
        return self.get_current_recommendations()

    def get_current_recommendations(self) -> List[Dict]:
        """Get current recommended products with their attributes"""
        return self.df.iloc[self.selected_indices].to_dict('records')

    def update_recommendations(self, liked_ids: List[str]) -> Dict:
        """Update recommendations based on user feedback"""
        if self.iteration >= 4:
            return {"error": "Maximum iterations reached"}

        self.iteration += 1
        
        # Update weights based on feedback
        selected_ids = self.ids[self.selected_indices]
        selected_X = self.X[self.selected_indices]
        liked_mask = np.isin(selected_ids, liked_ids)
        self.weights += selected_X[liked_mask].sum(axis=0)
        self.weights -= selected_X[~liked_mask].sum(axis=0)

        # Score and select new products
        scores = self.X.dot(self.weights) + np.random.randn(len(self.ids)) * 0.01
        self.selected_indices = np.argsort(scores)[-3:][::-1]

        # Generate prompt if this is the last iteration
        if self.iteration == 4:
            positive_attributes = [self.attributes[i] for i, w in enumerate(self.weights) if w > 0]
            prompt = "Looking for products with features: " + ", ".join(positive_attributes)
            return {
                "recommendations": self.get_current_recommendations(),
                "prompt": prompt,
                "iteration": self.iteration,
                "is_complete": True
            }

        return {
            "recommendations": self.get_current_recommendations(),
            "iteration": self.iteration,
            "is_complete": False
        }

    def reset(self) -> List[Dict]:
        """Reset the recommendation system"""
        self.weights = np.zeros(len(self.attributes))
        self.iteration = 0
        return self.initialize_recommendations()

# Pydantic models for API request/response
class FeedbackRequest(BaseModel):
    liked_ids: List[str]

class RecommendationResponse(BaseModel):
    recommendations: List[Dict]
    iteration: int
    is_complete: bool
    prompt: Optional[str] = None
    error: Optional[str] = None

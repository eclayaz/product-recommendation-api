# main.py

from fastapi import FastAPI, HTTPException
from product_recommendation import ProductRecommendation, FeedbackRequest, RecommendationResponse
from typing import List, Dict

app = FastAPI(title="Product Recommendation API")

# Create a global instance of the recommendation system
recommender = ProductRecommendation()

@app.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations():
    """Get current product recommendations"""
    try:
        recommendations = recommender.get_current_recommendations()
        return {
            "recommendations": recommendations,
            "iteration": recommender.iteration,
            "is_complete": recommender.iteration >= 4
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback", response_model=RecommendationResponse)
async def provide_feedback(feedback: FeedbackRequest):
    """Provide feedback on current recommendations and get new recommendations"""
    try:
        result = recommender.update_recommendations(feedback.liked_ids)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset", response_model=RecommendationResponse)
async def reset_recommendations():
    """Reset the recommendation system and get new initial recommendations"""
    try:
        recommendations = recommender.reset()
        return {
            "recommendations": recommendations,
            "iteration": 0,
            "is_complete": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
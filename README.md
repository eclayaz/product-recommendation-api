# Product Recommendation API

A FastAPI-based product recommendation system that uses user feedback to improve recommendations over time.

## Features

- Initial diverse product recommendations
- Iterative feedback-based recommendations
- Final product feature prompt generation
- RESTful API endpoints

## API Endpoints

- `GET /recommendations` - Get current product recommendations
- `POST /feedback` - Provide feedback on current recommendations
- `POST /reset` - Reset the recommendation system

## Setup

1. Clone the repository
2. Create a virtual environment: `python3 -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the server: `uvicorn main:app --reload`

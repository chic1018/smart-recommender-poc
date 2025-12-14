import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.recommend.contextual_frequency import router as contextual_router

# from backend.app.api.recommend.naive_bayes import router as naive_router

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-recommender-poc")

# App creation
app = FastAPI(title="Smart Recommender POC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # POC 阶段直接放开
    allow_credentials=True,
    allow_methods=["*"],  # 必须包含 OPTIONS
    allow_headers=["*"],
)

app.include_router(contextual_router)

# Development entrypoint
if __name__ == "__main__":
    # Run with reload for development; in production use an ASGI server manager
    # uvicorn backend.app.main:app --reload --port 8000
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)

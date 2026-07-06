from fastapi import FastAPI
from backend.api.v1.auth import router

app = FastAPI()
app.include_router(router)
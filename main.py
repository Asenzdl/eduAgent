import uvicorn

from fastapi import FastAPI
from backend.api.v1 import auth_router, resume_router, qa_router

app = FastAPI()
app.include_router(auth_router, prefix='/api/v1/auth')
app.include_router(resume_router, prefix='/api/v1/resume')
app.include_router(qa_router, prefix='/api/v1/qa')


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.endpoints import upload

app=FastAPI()
origins = [
    "http://localhost:3000", 
]

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Cho phép các domain trên
    allow_credentials=True,           # Cho phép cookies / Authorization header
    allow_methods=["*"],              # Cho phép tất cả các phương thức (GET, POST, PUT, DELETE, ...)
    allow_headers=["*"],              # Cho phép tất cả headers (Content-Type, Authorization, ...)
)
app.include_router(upload.router)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
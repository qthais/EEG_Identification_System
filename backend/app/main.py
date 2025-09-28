import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.endpoints import upload

# Tạo FastAPI app
app = FastAPI()

# Tạo Socket.IO server
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*"
)

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

# Include router
app.include_router(upload.router)

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    print(f"🔌 Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"❌ Client disconnected: {sid}")

# Tích hợp Socket.IO với FastAPI
socket_app = socketio.ASGIApp(sio, app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:socket_app", host="0.0.0.0", port=8000, reload=True)
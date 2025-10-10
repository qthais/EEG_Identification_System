import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.endpoints import upload, users

# Tạo FastAPI app
app = FastAPI()

# Tạo Socket.IO server
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=True,        # 👈 bật log Socket.IO
    engineio_logger=True  
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

# Include routers
app.include_router(upload.router, prefix="/api", tags=["EEG"])
app.include_router(users.router, prefix="/api", tags=["Users"])

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    print(f"🔌 Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"❌ Client disconnected: {sid}")

@sio.event
async def retrain_complete(sid, data):
    print(f"📡 Received retrain_complete from {sid}: {data}")
    # Forward cho tất cả client khác, trừ chính worker
    await sio.emit("retrain_complete", data, skip_sid=sid)

@sio.event
async def retrain_failed(sid, data):
    print(f"❌ Received retrain_failed from {sid}: {data}")
    # Forward cho tất cả client khác, trừ chính worker
    await sio.emit("retrain_failed", data, skip_sid=sid)

# Tích hợp Socket.IO với FastAPI
socket_app = socketio.ASGIApp(sio, app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:socket_app", host="0.0.0.0", port=8000, reload=True)
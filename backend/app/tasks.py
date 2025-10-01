import os
import socketio
from celery_app import celery_app
from app.services.retrain import retrainModel

# Create Socket.IO client to connect to the main server
sio = socketio.Client()
SOCKET_URL = os.getenv("SOCKET_URL", "http://localhost:8000")
@celery_app.task(name="tasks.retrain_model")
def retrain_model_task(filename: str):
    print('SOCKET_URL', SOCKET_URL)
    try:
        print(f"🔁 Celery: retraining for {filename}")
        print("Worker sees uploads:", os.listdir("/app/app/data/uploads"))
        test_accuracy = retrainModel()

        # 🔹 Emit event after retrain
        try:
            sio.connect(SOCKET_URL)
            sio.emit("retrain_complete", {
                "filename": filename,
                "test_accuracy": test_accuracy
            })
            print("📡 Event retrain_complete emitted")
            sio.disconnect()
        except Exception as e:
            print(f"❌ Socket.IO connection error: {e}")

        return {"success": True, "accuracy": test_accuracy}

    except Exception as e:
        try:
            sio.connect(SOCKET_URL)
            sio.emit("retrain_failed", {
                "filename": filename,
                "error": str(e)
            })
            sio.disconnect()
        except Exception as conn_error:
            print(f"❌ Socket.IO connection error: {conn_error}")

        return {"success": False, "error": str(e)}

from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import JSONResponse
from app.auth.token_utils import decode_token
from database.db import prediction_collection
from database.models import PredictionResponse
from fastapi.encoders import jsonable_encoder
router = APIRouter()

@router.get("/predictions")
async def get_predictions_by_user(Authorization: str = Header(None)):
    """
    Return all prediction records associated with a given user_id.
    """
    if not Authorization or not Authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = Authorization.split(" ")[1]

    # ✅ Decode token using your existing helper
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = payload.get("sub")
    user_name = payload.get("name")

    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing user_id")
    cursor = prediction_collection.find({"user_id": user_id})
    predictions = await cursor.to_list(length=None)

    if not predictions:
        return JSONResponse(
            {"success": False, "message": f"No predictions found for user_id '{user_id}'"},
            status_code=404
        )


    for p in predictions:
        p.pop("_id", None)

    return jsonable_encoder({
        "success": True,
        "count": len(predictions),
        "user_id": user_id,
        "user_name": user_name,
        "predictions": predictions
    })


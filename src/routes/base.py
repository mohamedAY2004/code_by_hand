from fastapi import APIRouter
base_router = APIRouter()

@base_router.get("/")
async def health_check():
    return {"message": "Hello, World!"}
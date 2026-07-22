from pydantic import BaseModel, EmailStr

class UpdateUserRequest(BaseModel):
    username: str
    email: EmailStr
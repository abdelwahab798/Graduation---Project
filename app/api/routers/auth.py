from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from Database.config import get_db
import Database.models as models
from pydantic import BaseModel, EmailStr, Field
import os 
from dotenv import load_dotenv
import bcrypt
from sqlalchemy.orm import joinedload
from typing import Optional

load_dotenv()

SECRET_KEY = os.getenv("secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120 

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str = None
    role: str = Field(..., description="يجب أن يكون إما 'patient' أو 'doctor'")
    
    
    age: Optional[float] = None
    gender: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str



def hash_password(password: str) -> str:
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    plain_bytes = plain_password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(plain_bytes, hashed_bytes)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
        
    user = db.query(models.User).filter(
    (models.User.username == username) | (models.User.email == username)
).filter(models.User.username == username).first()
    if user is None:
        raise credentials_exception
    return user



def verify_is_doctor(current_user: models.User = Depends(get_current_user)):
   
    if current_user.role != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="عذراً، الأطباء فقط يملكون هذه الصلاحية."
        )
    return current_user




# 1. التسجيل الذكي 
@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(user_data: UserRegister, db: Session = Depends(get_db)):
    if user_data.role not in ["patient", "doctor"]:
        raise HTTPException(status_code=400, detail="الـ Role يجب أن يكون إما 'patient' أو 'doctor'")
        
    db_user = db.query(models.User).filter(
        (models.User.username == user_data.username) | (models.User.email == user_data.email)
    ).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username or Email already registered")
    
    try:
       
        new_user = models.User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hash_password(user_data.password),
            full_name=user_data.full_name,
            role=user_data.role
        )
        db.add(new_user)
        db.flush() 
        
      
        if user_data.role == "patient":
            if user_data.age is None or user_data.gender is None:
                raise HTTPException(
                    status_code=400, 
                    detail="السن (age) والجنس (gender) حقول إجبارية عند تسجيل المريض."
                )
            
            new_patient = models.Patient(
                user_id=new_user.user_id,
                age=user_data.age,
                gender=user_data.gender
            )
            db.add(new_patient)
        
        db.commit()
        return {"message": f"Account successfully registered as {user_data.role}"}
        
    except HTTPException as http_ex:
        db.rollback()
        raise http_ex
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error during registration: {str(e)}")


# 2. تسجيل الدخول (حقن الـ Role جوه الـ JWT لراحة الـ Front-end)
@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(
        (models.User.email == form_data.username) | (models.User.username == form_data.username)
    ).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token = create_access_token(
        data={
            "sub": user.username,
            "role": user.role,
            "user_id": user.user_id
        }
    )
    return {"access_token": access_token, "token_type": "bearer"}
import asyncio
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import or_, select

from app.core.logging_db import log_event

from app.core.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    refresh_token_expiry,
    verify_password,
)
from app.deps import DB, CurrentUser
from app.models import RefreshToken, User
from app.schemas import (
    AccessToken,
    LoginRequest,
    Message,
    RefreshRequest,
    TokenPair,
    UserCreate,
    UserRead,
)

router = APIRouter(prefix="/auth", tags=["auth"])


async def _authenticate(db, identifier: str, password: str) -> User:
    result = await db.execute(
        select(User).where(
            or_(User.username == identifier, User.email == identifier)
        )
    )
    user = result.scalar_one_or_none()
    if user is None or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user"
        )
    return user


async def _issue_tokens(db, user: User) -> TokenPair:
    raw_refresh, token_hash = generate_refresh_token()
    db.add(
        RefreshToken(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=refresh_token_expiry(),
        )
    )
    await db.commit()
    asyncio.create_task(log_event(
        "user_login", user_id=user.id, entity_type="user", entity_id=user.id))
    return TokenPair(
        access_token=create_access_token(user.id),
        refresh_token=raw_refresh,
    )


@router.post(
    "/register", response_model=UserRead, status_code=status.HTTP_201_CREATED
)
async def register(payload: UserCreate, db: DB):
    exists = await db.execute(
        select(User).where(
            or_(User.username == payload.username, User.email == payload.email)
        )
    )
    if exists.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email already registered",
        )
    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    asyncio.create_task(log_event(
        "user_registered", user_id=user.id, entity_type="user", entity_id=user.id,
        details={"username": user.username}))
    return user


@router.post("/login", response_model=TokenPair)
async def login(
    db: DB,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
):
    """OAuth2 password flow. `username` may be a username or an email."""
    user = await _authenticate(db, form_data.username, form_data.password)
    return await _issue_tokens(db, user)


@router.post("/login/json", response_model=TokenPair)
async def login_json(payload: LoginRequest, db: DB):
    user = await _authenticate(db, payload.username_or_email, payload.password)
    return await _issue_tokens(db, user)


@router.post("/refresh", response_model=AccessToken)
async def refresh(payload: RefreshRequest, db: DB):
    token_hash = hash_refresh_token(payload.refresh_token)
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    token = result.scalar_one_or_none()
    now = datetime.now(timezone.utc)
    if (
        token is None
        or token.revoked
        or token.expires_at.replace(tzinfo=timezone.utc) < now
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    return AccessToken(access_token=create_access_token(token.user_id))


@router.post("/refresh/rotate", response_model=TokenPair)
async def refresh_rotate(payload: RefreshRequest, db: DB):
    """Exchange a refresh token for a new pair, revoking the old token."""
    token_hash = hash_refresh_token(payload.refresh_token)
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    token = result.scalar_one_or_none()
    now = datetime.now(timezone.utc)
    if (
        token is None
        or token.revoked
        or token.expires_at.replace(tzinfo=timezone.utc) < now
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    token.revoked = True
    user = await db.get(User, token.user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User unavailable"
        )
    return await _issue_tokens(db, user)


@router.post("/logout", response_model=Message)
async def logout(payload: RefreshRequest, db: DB):
    """Revoke a single refresh token."""
    token_hash = hash_refresh_token(payload.refresh_token)
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    token = result.scalar_one_or_none()
    if token is not None and not token.revoked:
        token.revoked = True
        await db.commit()
    return Message(detail="Logged out")


@router.post("/logout/all", response_model=Message)
async def logout_all(db: DB, current_user: CurrentUser):
    """Revoke all of the current user's active refresh tokens."""
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.user_id == current_user.id,
            RefreshToken.revoked.is_(False),
        )
    )
    for token in result.scalars().all():
        token.revoked = True
    await db.commit()
    return Message(detail="All sessions revoked")


@router.get("/me", response_model=UserRead)
async def me(current_user: CurrentUser):
    return current_user

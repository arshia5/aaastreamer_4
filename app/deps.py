import logging
import os
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Query, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import decode_access_token
from app.models import User

# --- DEV ONLY: bypass authentication --------------------------------------- #
# Set DEV_NO_AUTH=true to run with no auth: every request is treated as a logged-in
# user (an admin if one exists, or DEV_USER_ID if set). Token becomes optional.
# Production is unaffected when the var is unset. NEVER enable this in production.
DEV_NO_AUTH = os.getenv("DEV_NO_AUTH", "").lower() in ("1", "true", "yes", "on")
DEV_USER_ID = os.getenv("DEV_USER_ID")  # optional: pin the dev user to a specific id

if DEV_NO_AUTH:
    logging.getLogger("app.deps").warning(
        "DEV_NO_AUTH is ON — authentication is DISABLED. Do not use in production.")

# auto_error off in dev so a missing Authorization header doesn't 401 before our
# dependency runs (the token is then ignored).
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=not DEV_NO_AUTH)

DB = Annotated[AsyncSession, Depends(get_db)]


async def get_current_user(
    db: DB,
    token: Annotated[str | None, Depends(oauth2_scheme)],
) -> User:
    if DEV_NO_AUTH:
        if DEV_USER_ID:
            user = await db.get(User, int(DEV_USER_ID))
        else:
            user = await db.scalar(
                select(User).order_by(User.is_admin.desc(), User.id).limit(1))
        if user is not None:
            return user
        # no users in the DB yet — fall through to normal (token-based) handling

    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        if payload.get("type") != "access":
            raise credentials_exc
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise credentials_exc

    user = await db.get(User, user_id)
    if user is None:
        raise credentials_exc
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user"
        )
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_current_admin(user: CurrentUser) -> User:
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return user


CurrentAdmin = Annotated[User, Depends(get_current_admin)]


class Pagination:
    def __init__(
        self,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        self.limit = limit
        self.offset = offset


PageParams = Annotated[Pagination, Depends(Pagination)]

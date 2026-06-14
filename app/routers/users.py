from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError

from app.core.security import hash_password
from app.deps import DB, CurrentAdmin, CurrentUser, PageParams
from app.models import User
from app.schemas import Message, UserCreate, UserRead, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[UserRead])
async def list_users(
    db: DB,
    page: PageParams,
    _: CurrentAdmin,
    search: str | None = Query(default=None),
    is_active: bool | None = Query(default=None),
):
    stmt = select(User)
    if search:
        like = f"%{search}%"
        stmt = stmt.where(or_(User.username.ilike(like), User.email.ilike(like)))
    if is_active is not None:
        stmt = stmt.where(User.is_active.is_(is_active))
    stmt = stmt.order_by(User.id).limit(page.limit).offset(page.offset)
    return (await db.execute(stmt)).scalars().all()


@router.get("/count")
async def count_users(db: DB, _: CurrentAdmin):
    total = await db.scalar(select(func.count()).select_from(User))
    return {"count": total}


@router.get("/me", response_model=UserRead)
async def read_me(current_user: CurrentUser):
    return current_user


@router.patch("/me", response_model=UserRead)
async def update_me(payload: UserUpdate, db: DB, current_user: CurrentUser):
    data = payload.model_dump(exclude_unset=True)
    # Self-service cannot change privilege/active flags.
    data.pop("is_admin", None)
    data.pop("is_active", None)
    return await _apply_user_update(db, current_user, data)


@router.post("", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(payload: UserCreate, db: DB, _: CurrentAdmin):
    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409, detail="Username or email already exists"
        )
    await db.refresh(user)
    return user


@router.get("/{user_id}", response_model=UserRead)
async def get_user(user_id: int, db: DB, _: CurrentAdmin):
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.patch("/{user_id}", response_model=UserRead)
async def update_user(
    user_id: int, payload: UserUpdate, db: DB, _: CurrentAdmin
):
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    data = payload.model_dump(exclude_unset=True)
    return await _apply_user_update(db, user, data)


@router.delete("/{user_id}", response_model=Message)
async def delete_user(user_id: int, db: DB, _: CurrentAdmin):
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    await db.delete(user)
    await db.commit()
    return Message(detail="User deleted")


async def _apply_user_update(db, user: User, data: dict) -> User:
    if "password" in data:
        user.password_hash = hash_password(data.pop("password"))
    for field, value in data.items():
        setattr(user, field, value)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409, detail="Username or email already exists"
        )
    await db.refresh(user)
    return user

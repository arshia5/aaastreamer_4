"""Create (or promote) an admin user.

Usage:
    python -m scripts.create_admin <username> <email> <password>
"""
import asyncio
import sys

from sqlalchemy import or_, select

from app.core.database import AsyncSessionLocal
from app.core.security import hash_password
from app.models import User


async def main(username: str, email: str, password: str) -> None:
    async with AsyncSessionLocal() as db:
        existing = (
            await db.execute(
                select(User).where(
                    or_(User.username == username, User.email == email)
                )
            )
        ).scalar_one_or_none()
        if existing:
            existing.is_admin = True
            existing.is_active = True
            existing.password_hash = hash_password(password)
            print(f"Updated existing user '{existing.username}' -> admin")
        else:
            db.add(
                User(
                    username=username,
                    email=email,
                    password_hash=hash_password(password),
                    is_admin=True,
                    is_active=True,
                )
            )
            print(f"Created admin user '{username}'")
        await db.commit()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3]))

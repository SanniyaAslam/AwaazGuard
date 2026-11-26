"""Database models. SQLite by default (free, no server process); any SQLAlchemy URL works via DATABASE_URL."""
from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text,
                        create_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./awaazguard_local.db")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    variant: Mapped[str] = mapped_column(String(16), index=True)
    model_version: Mapped[str] = mapped_column(String(32))
    prob_fake: Mapped[float] = mapped_column(Float)
    margin: Mapped[float] = mapped_column(Float)
    verdict: Mapped[str] = mapped_column(String(16))
    duration_s: Mapped[float] = mapped_column(Float)
    rms: Mapped[float] = mapped_column(Float)
    clipped_fraction: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float)
    filename: Mapped[str] = mapped_column(String(255), default="")
    audio_key: Mapped[str] = mapped_column(String(512), default="")
    segments: Mapped[list] = mapped_column(JSON, default=list)
    embedding: Mapped[list] = mapped_column(JSON, default=list)


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True)
    label: Mapped[str] = mapped_column(String(16))           # bonafide | spoof
    source: Mapped[str] = mapped_column(String(32), default="user")
    exported: Mapped[bool] = mapped_column(Boolean, default=False)


class DriftRun(Base):
    __tablename__ = "drift_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    model_version: Mapped[str] = mapped_column(String(32))
    n_current: Mapped[int] = mapped_column(Integer)
    mmd: Mapped[float] = mapped_column(Float)
    mmd_p_value: Mapped[float] = mapped_column(Float)
    ood_rate: Mapped[float] = mapped_column(Float)
    drifted_features_share: Mapped[float] = mapped_column(Float)
    drift_detected: Mapped[bool] = mapped_column(Boolean)
    report_path: Mapped[str] = mapped_column(Text, default="")


class CanaryEvent(Base):
    __tablename__ = "canary_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    action: Mapped[str] = mapped_column(String(32))          # start | promote | rollback | weight
    weight: Mapped[float] = mapped_column(Float)
    detail: Mapped[str] = mapped_column(Text, default="")


_engine = create_engine(DATABASE_URL, pool_pre_ping=True,
                        connect_args={"check_same_thread": False, "timeout": 30} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(_engine)

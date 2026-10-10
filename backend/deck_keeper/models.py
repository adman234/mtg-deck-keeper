from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class Group(Base):
    __tablename__ = "groups"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    full_name: Mapped[str] = mapped_column(String(255), default="")
    password_hash: Mapped[str | None] = mapped_column(String(255))
    auth_method: Mapped[str] = mapped_column(String(20), default="local")  # local | oidc
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Setting(Base):
    """Runtime settings edited in the GUI, plus internal state under the ``state.`` prefix."""

    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


class Card(Base):
    """One row per Oracle card name. Details are filled lazily from Scryfall."""

    __tablename__ = "cards"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    oracle_id: Mapped[str | None] = mapped_column(String(64))
    type_line: Mapped[str | None] = mapped_column(String(255))
    mana_cost: Mapped[str | None] = mapped_column(String(100))
    cmc: Mapped[float | None] = mapped_column(Float)
    colors: Mapped[str | None] = mapped_column(String(10))
    image_uri: Mapped[str | None] = mapped_column(String(500))
    scryfall_uri: Mapped[str | None] = mapped_column(String(500))
    details_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Deck(Base):
    __tablename__ = "decks"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    format: Mapped[str] = mapped_column(String(40), default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    cards: Mapped[list["DeckCard"]] = relationship(cascade="all, delete-orphan", back_populates="deck")


class DeckCard(Base):
    __tablename__ = "deck_cards"
    __table_args__ = (UniqueConstraint("deck_id", "card_id", "board"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    deck_id: Mapped[int] = mapped_column(ForeignKey("decks.id", ondelete="CASCADE"))
    card_id: Mapped[int] = mapped_column(ForeignKey("cards.id"))
    qty: Mapped[int] = mapped_column(Integer, default=1)
    board: Mapped[str] = mapped_column(String(20), default="main")  # main | side | commander | maybe
    deck: Mapped[Deck] = relationship(back_populates="cards")
    card: Mapped[Card] = relationship()


class ScanJob(Base):
    __tablename__ = "scan_jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued|processing|review|saved|failed
    prepass: Mapped[bool] = mapped_column(Boolean, default=True)
    model: Mapped[str] = mapped_column(String(200), default="")
    step: Mapped[str] = mapped_column(String(200), default="")
    error: Mapped[str | None] = mapped_column(Text)
    total_images: Mapped[int] = mapped_column(Integer, default=0)
    processed_images: Mapped[int] = mapped_column(Integer, default=0)
    deck_id: Mapped[int | None] = mapped_column(ForeignKey("decks.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    images: Mapped[list["ScanImage"]] = relationship(cascade="all, delete-orphan", order_by="ScanImage.id")
    lines: Mapped[list["ScanLine"]] = relationship(cascade="all, delete-orphan", order_by="ScanLine.id")


class ScanImage(Base):
    __tablename__ = "scan_images"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("scan_jobs.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(String(500))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    crops_found: Mapped[int] = mapped_column(Integer, default=0)
    detections: Mapped[list["Detection"]] = relationship(cascade="all, delete-orphan", order_by="Detection.id")


class Detection(Base):
    """Raw evidence: one thing the model read, from a whole photo or a single-card crop."""

    __tablename__ = "detections"
    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("scan_images.id", ondelete="CASCADE"))
    source: Mapped[str] = mapped_column(String(10))  # crop | photo
    crop_path: Mapped[str | None] = mapped_column(String(500))
    bbox: Mapped[list | None] = mapped_column(JSON)
    raw_name: Mapped[str] = mapped_column(String(255))
    raw_set: Mapped[str | None] = mapped_column(String(20))
    raw_number: Mapped[str | None] = mapped_column(String(20))
    qty: Mapped[int] = mapped_column(Integer, default=1)
    model_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    card_id: Mapped[int | None] = mapped_column(ForeignKey("cards.id"))
    match_score: Mapped[float] = mapped_column(Float, default=0.0)


class ScanLine(Base):
    """What the user reviews and edits: one card + quantity, aggregated from detections."""

    __tablename__ = "scan_lines"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("scan_jobs.id", ondelete="CASCADE"))
    card_id: Mapped[int | None] = mapped_column(ForeignKey("cards.id"))
    raw_name: Mapped[str] = mapped_column(String(255), default="")
    qty: Mapped[int] = mapped_column(Integer, default=1)
    board: Mapped[str] = mapped_column(String(20), default="main")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    flags: Mapped[list] = mapped_column(JSON, default=list)
    suggestions: Mapped[list] = mapped_column(JSON, default=list)  # [{card_id, name, score}]
    detection_ids: Mapped[list] = mapped_column(JSON, default=list)
    accepted: Mapped[bool] = mapped_column(Boolean, default=False)
    user_edited: Mapped[bool] = mapped_column(Boolean, default=False)
    card: Mapped[Card | None] = relationship()

"""Popula o banco com o dataset Steam já limpo por ``recsys.data``.

Uso:
    python scripts/seed_db.py [--data-dir data/raw/steam]

As tabelas precisam existir antes (``alembic upgrade head``).

O seed é idempotente: registros que já existem (mesma chave ou nome de jogo) são
ignorados, então rodar de novo não duplica nada nem sobrescreve alterações feitas pela API.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from sqlalchemy import func, insert, inspect, select, text
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.models import Game, Interaction, User  # noqa: E402
from app.db.session import engine as default_engine  # noqa: E402
from recsys.data import DEFAULT_DATA_DIR, Dataset, load_dataset  # noqa: E402

BATCH_SIZE = 10_000
REQUIRED_TABLES = {"users", "games", "interactions"}


def _batches(rows: list[dict], size: int = BATCH_SIZE):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _insert_ignoring_existing(session: Session, model):
    """``INSERT ... ON CONFLICT DO NOTHING`` no PostgreSQL e no SQLite."""
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        return postgresql.insert(model).on_conflict_do_nothing()
    if dialect == "sqlite":
        return sqlite.insert(model).on_conflict_do_nothing()
    return insert(model)


def _count(session: Session, model) -> int:
    return session.scalar(select(func.count()).select_from(model))


def _reset_sequences(session: Session) -> None:
    """No PostgreSQL, avança as sequências para depois dos IDs vindos do dataset."""
    if session.get_bind().dialect.name != "postgresql":
        return
    for table in ("users", "games"):
        session.execute(
            text(
                f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                f"COALESCE((SELECT MAX(id) FROM {table}), 0) + 1, false)"
            )
        )


def seed(engine: Engine, dataset: Dataset) -> dict[str, int]:
    """Insere usuários, jogos e interações em uma única transação, sem duplicar.

    Returns:
        Quantos registros novos entraram em cada tabela.
    """
    missing = REQUIRED_TABLES - set(inspect(engine).get_table_names())
    if missing:
        raise RuntimeError(
            f"Tabelas ausentes: {', '.join(sorted(missing))}. Rode 'alembic upgrade head'."
        )

    users = [
        {"id": int(r.user_id), "steam_real_id": int(r.steam_real_id)}
        for r in dataset.users.itertuples(index=False)
    ]
    games = [
        {"id": int(r.game_id), "name": r.name, "tags": list(r.tags)}
        for r in dataset.games.itertuples(index=False)
    ]
    interactions = [
        {
            "user_id": int(r.user_id),
            "game_id": int(r.game_id),
            "purchased": True,
            "hours": float(r.hours),
            "disliked": False,
        }
        for r in dataset.interactions.itertuples(index=False)
    ]

    inserted = {}
    with Session(engine) as session, session.begin():
        for name, model, rows in (
            ("users", User, users),
            ("games", Game, games),
            ("interactions", Interaction, interactions),
        ):
            before = _count(session, model)
            stmt = _insert_ignoring_existing(session, model)
            for batch in _batches(rows):
                session.execute(stmt, batch)
            inserted[name] = _count(session, model) - before
        _reset_sequences(session)

    return inserted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Popula o banco com o dataset Steam.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="pasta com os .dat")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    counts = seed(default_engine, load_dataset(args.data_dir))
    elapsed = time.perf_counter() - start
    summary = ", ".join(f"{n} {name}" for name, n in counts.items())
    if not any(counts.values()):
        print(f"Banco já estava populado; nada a inserir ({elapsed:.1f}s).")
    else:
        print(f"Seed concluído em {elapsed:.1f}s; novos registros: {summary}.")


if __name__ == "__main__":
    main()

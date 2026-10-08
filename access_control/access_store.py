from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3


class AccessStore:
    """Visits survive tracking changes and restarts; each crossing is audited."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=5)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS acessos (
                    id INTEGER PRIMARY KEY,
                    nome TEXT NOT NULL,
                    horario_entrada TEXT NOT NULL,
                    horario_saida TEXT,
                    tempo_permanencia_segundos REAL,
                    execucao_entrada TEXT NOT NULL,
                    tracking_entrada INTEGER NOT NULL,
                    execucao_saida TEXT,
                    tracking_saida INTEGER
                );
                CREATE UNIQUE INDEX IF NOT EXISTS uma_visita_aberta_por_pessoa
                    ON acessos(nome) WHERE horario_saida IS NULL;
                CREATE TABLE IF NOT EXISTS cruzamentos (
                    id INTEGER PRIMARY KEY,
                    execucao TEXT NOT NULL,
                    tracking_id INTEGER NOT NULL,
                    direcao TEXT NOT NULL CHECK(direcao IN ('entrada', 'saida')),
                    horario TEXT NOT NULL,
                    nome TEXT,
                    reconhecido_no_cruzamento INTEGER NOT NULL,
                    acesso_id INTEGER REFERENCES acessos(id),
                    acao TEXT NOT NULL
                );
            """)
        except BaseException:
            self.connection.close()
            raise
        self.open_count = self.connection.execute(
            "SELECT COUNT(*) FROM acessos WHERE horario_saida IS NULL"
        ).fetchone()[0]

    def register_crossing(self, run_id: str, track_id: int, direction: str,
                          timestamp: datetime, name: str | None) -> dict:
        if timestamp.utcoffset() is None:
            raise ValueError("O horario precisa informar o fuso")
        with self.connection:
            cursor = self.connection.execute("""
                INSERT INTO cruzamentos
                    (execucao, tracking_id, direcao, horario, nome,
                     reconhecido_no_cruzamento, acao)
                VALUES (?, ?, ?, ?, ?, ?, 'desconhecido')
            """, (run_id, track_id, direction, timestamp.isoformat(), name, int(name is not None)))
            event_id = cursor.lastrowid
            if name is not None:
                self._apply_visit(event_id)
        return self.get_crossing(event_id)

    def identify_crossing(self, event_id: int, name: str) -> dict:
        """Resolve a pending crossing using its original timestamp, exactly once."""
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE cruzamentos SET nome=? WHERE id=? AND nome IS NULL",
                (name, event_id),
            )
            if cursor.rowcount:
                self._apply_visit(event_id)
        return self.get_crossing(event_id)

    def get_crossing(self, event_id: int) -> dict:
        row = self.connection.execute("SELECT * FROM cruzamentos WHERE id=?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(event_id)
        return dict(row)

    def _apply_visit(self, event_id: int):
        event = self.get_crossing(event_id)
        visit = self.connection.execute(
            "SELECT * FROM acessos WHERE nome=? AND horario_saida IS NULL", (event["nome"],)
        ).fetchone()
        visit_id = visit["id"] if visit else None
        if event["direcao"] == "entrada":
            action = "entrada_ja_aberta"
            if visit is None:
                cursor = self.connection.execute("""
                    INSERT INTO acessos (nome, horario_entrada, execucao_entrada, tracking_entrada)
                    VALUES (?, ?, ?, ?)
                """, (event["nome"], event["horario"], event["execucao"], event["tracking_id"]))
                visit_id = cursor.lastrowid
                self.open_count += 1
                action = "entrada_registrada"
        else:
            action = "saida_sem_entrada"
            if visit is not None:
                seconds = (datetime.fromisoformat(event["horario"]) -
                           datetime.fromisoformat(visit["horario_entrada"])).total_seconds()
                if seconds >= 0:
                    self.connection.execute("""
                        UPDATE acessos SET horario_saida=?, tempo_permanencia_segundos=?,
                            execucao_saida=?, tracking_saida=? WHERE id=?
                    """, (event["horario"], round(seconds, 3), event["execucao"],
                          event["tracking_id"], visit_id))
                    self.open_count -= 1
                    action = "saida_registrada"
                else:
                    action = "horario_invalido"
        self.connection.execute(
            "UPDATE cruzamentos SET acesso_id=?, acao=? WHERE id=?", (visit_id, action, event_id)
        )

    def close(self):
        self.connection.close()

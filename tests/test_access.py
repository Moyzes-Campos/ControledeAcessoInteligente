from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np

from access_control.access import AccessController
from access_control.access_store import AccessStore
from access_control.line_crossing import LineCounter
from access_control.types import TrackIdentity, TrackedPose


START = datetime(2026, 10, 6, 10, 0, tzinfo=timezone(timedelta(hours=-3)))


def track(track_id, center_y):
    return TrackedPose(track_id, 0, np.array([40, center_y - 10, 60, center_y + 10]), .9, False)


class AccessTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "acessos.sqlite3"
        self.store = AccessStore(self.path)
        self.addCleanup(lambda: self.store.close())
        self.control = AccessController(LineCounter((0, 50), (100, 50), hysteresis=2), self.store)

    def update(self, track_id, center_y, known=True, seconds=0, name="Moyzes"):
        identity = TrackIdentity(label=name, known=known)
        return self.control.update([track(track_id, center_y)], {track_id: identity}, {track_id},
                                   START + timedelta(seconds=seconds), monotonic_time=seconds)

    def visits(self):
        return self.store.connection.execute("SELECT * FROM acessos ORDER BY id").fetchall()

    def enter(self):
        self.update(1, 30)
        return self.update(1, 70, seconds=10)

    def test_new_tracking_closes_same_visit_and_computes_duration(self):
        state = self.enter()
        self.assertEqual(state.events[0]["acao"], "entrada_registrada")
        # Losing a tracking does not represent an exit.
        state = self.control.update([], {}, set(), START + timedelta(seconds=20), 20)
        self.assertEqual((state.lamp, state.open_visits), ("orange", 1))
        self.update(8, 70, seconds=100)
        state = self.update(8, 30, seconds=130)
        rows = self.visits()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["tracking_entrada"], rows[0]["tracking_saida"]), (1, 8))
        self.assertEqual(rows[0]["tempo_permanencia_segundos"], 120.0)
        self.assertEqual(rows[0]["horario_saida"], (START + timedelta(seconds=130)).isoformat())
        self.assertEqual((state.entries, state.exits, state.open_visits), (1, 1, 0))

    def test_unknown_crossing_counts_and_sounds_without_named_visit(self):
        self.update(2, 30, known=False)
        state = self.update(2, 70, known=False, seconds=10)
        self.assertEqual((state.lamp, state.entries, state.alarm_until), ("red", 1, 13.0))
        self.assertEqual(self.visits(), [])
        event = state.events[0]
        self.assertIsNone(event["nome"])
        self.assertEqual(event["reconhecido_no_cruzamento"], 0)

    def test_late_recognition_uses_original_crossing_time_once(self):
        self.update(2, 30, known=False)
        crossing = self.update(2, 70, known=False, seconds=10).events[0]
        state = self.update(2, 75, known=True, seconds=12)
        self.assertEqual(state.lamp, "green")
        self.assertEqual(state.events[0]["id"], crossing["id"])
        self.assertEqual(self.visits()[0]["horario_entrada"], (START + timedelta(seconds=10)).isoformat())
        self.assertEqual(self.update(2, 80, seconds=14).events, [])
        self.assertEqual(len(self.visits()), 1)

    def test_exit_without_entry_is_audited_without_fabricating_visit(self):
        self.update(3, 70)
        state = self.update(3, 30, seconds=10)
        self.assertEqual(state.events[0]["acao"], "saida_sem_entrada")
        self.assertEqual(self.visits(), [])

    def test_duplicate_entry_does_not_overwrite_first_entry(self):
        self.enter()
        self.update(9, 30, seconds=15)
        state = self.update(9, 70, seconds=20)
        self.assertEqual(state.events[0]["acao"], "entrada_ja_aberta")
        self.assertEqual(len(self.visits()), 1)
        self.assertEqual(self.visits()[0]["horario_entrada"], (START + timedelta(seconds=10)).isoformat())

    def test_second_visit_after_exit_creates_new_row(self):
        self.enter()
        self.update(1, 30, seconds=20)
        self.update(1, 70, seconds=30)
        self.assertEqual(len(self.visits()), 2)
        self.assertIsNone(self.visits()[1]["horario_saida"])

    def test_open_visit_survives_restart_and_track_id_reuse(self):
        original_run = self.control.run_id
        self.enter()
        self.store.close()
        self.store = AccessStore(self.path)
        self.control = AccessController(LineCounter((0, 50), (100, 50), 2), self.store)
        self.update(1, 70, seconds=40)
        self.update(1, 30, seconds=70)
        row = self.visits()[0]
        self.assertEqual(row["tempo_permanencia_segundos"], 60)
        self.assertEqual(row["execucao_entrada"], original_run)
        self.assertNotEqual(row["execucao_entrada"], row["execucao_saida"])

    def test_unknown_has_priority_when_multiple_people_are_visible(self):
        state = self.control.update([track(1, 30), track(2, 30)],
                                    {1: TrackIdentity("Moyzes", known=True), 2: TrackIdentity()},
                                    {1, 2}, START, 0)
        self.assertEqual(state.lamp, "red")

    def test_unknown_exit_also_triggers_alarm(self):
        self.update(2, 70, known=False)
        state = self.update(2, 30, known=False, seconds=10)
        self.assertEqual(state.exits, 1)
        self.assertEqual(state.alarm_until, 13)

    def test_center_crossing_counts_when_box_base_is_outside_segment(self):
        self.control.counter = LineCounter((50, 0), (50, 60), hysteresis=2)
        for seconds, x in enumerate((30, 70)):
            person = TrackedPose(2, 0, np.array([x - 10, 10, x + 10, 90]), .9, False)
            state = self.control.update([person], {2: TrackIdentity()}, {2},
                                        START + timedelta(seconds=seconds), seconds)
        self.assertEqual((state.entries, state.exits), (0, 1))
        self.assertEqual(state.alarm_until, 4)
        self.assertEqual(len(state.events), 1)

    def test_lost_unknown_does_not_assign_pending_event_to_reused_id(self):
        self.update(2, 30, known=False)
        self.update(2, 70, known=False, seconds=10)
        self.control.update([], {}, set(), START, 12)
        self.update(2, 70, seconds=15)
        self.assertEqual(self.visits(), [])


if __name__ == "__main__":
    unittest.main()

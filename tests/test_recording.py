import unittest
from unittest.mock import Mock

from access_control.recording import RealtimeVideoWriter


class RecordingTests(unittest.TestCase):
    def test_variable_updates_preserve_elapsed_duration_and_displayed_frames(self):
        writer = Mock()
        clock = Mock(side_effect=[10.0, 10.2, 10.5, 11.0])
        recording = RealtimeVideoWriter(writer, 30, clock)
        recording.write('a')
        recording.write('b')
        recording.write('c')
        recording.release()
        frames = [call.args[0] for call in writer.write.call_args_list]
        self.assertEqual(len(frames), 30)
        self.assertEqual(frames, ['a'] * 6 + ['b'] * 9 + ['c'] * 15)
        writer.release.assert_called_once()

    def test_empty_recording_releases_writer(self):
        writer = Mock()
        RealtimeVideoWriter(writer, 30).release()
        writer.write.assert_not_called()
        writer.release.assert_called_once()

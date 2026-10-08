from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import unittest

from access_control.esp32 import EspGpioClient


class Esp32Tests(unittest.TestCase):
    def setUp(self):
        self.commands = []
        self.levels = {}
        self.condition = threading.Condition()
        self.fail = False
        self.ignore_gate_commands = 0
        test = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                action = payload['gate_action']
                with test.condition:
                    failed = test.fail
                    test.commands.append((1, action))
                    if not failed:
                        if test.ignore_gate_commands:
                            test.ignore_gate_commands -= 1
                        else:
                            test.levels[1] = action == 'on'
                    state = 'on' if test.levels.get(1) else 'off'
                    test.condition.notify_all()
                self.send_response(403 if failed else 200)
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error' if failed else 'ok',
                                             'gate_state': state, 'gate_enabled': True}).encode())

            def do_GET(self):
                pin, action = self.path.removeprefix("/gpio").split(":")
                with test.condition:
                    failed = test.fail
                    test.commands.append((int(pin), action))
                    if not failed:
                        test.levels[int(pin)] = action == "on"
                    test.condition.notify_all()
                self.send_response(403 if failed else 200)
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error" if failed else "ok"}).encode())

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.client = EspGpioClient(f"http://127.0.0.1:{self.server.server_port}",
                                    timeout=.3, heartbeat=.4, retry_interval=.05)
        self.addCleanup(self.client.close)
        self.wait_for({1: False, 4: False, 5: True, 6: False, 7: False})

    def wait_for(self, levels):
        with self.condition:
            self.assertTrue(self.condition.wait_for(
                lambda: all(self.levels.get(pin) == value for pin, value in levels.items()), timeout=3
            ), f"Esperado {levels}, recebido {self.levels}")

    def test_led_transitions_turn_previous_lamp_off_first(self):
        self.client.update("red")
        self.wait_for({5: False, 6: True, 7: False})
        self.client.update("green")
        self.wait_for({5: False, 6: False, 7: True})
        with self.condition:
            commands = self.commands.copy()
        self.assertLess(commands.index((5, "off")), commands.index((6, "on")))
        self.assertLess(commands.index((6, "off"), commands.index((6, "on"))),
                        commands.index((7, "on")))

    def test_buzzer_expires_without_another_video_frame(self):
        self.client.update("red", time.monotonic() + .25)
        self.wait_for({4: True, 6: True})
        self.wait_for({4: False, 6: True})

    def test_close_returns_to_orange_and_turns_buzzer_off(self):
        self.client.update("green", time.monotonic() + 10)
        self.wait_for({1: True, 4: True, 7: True})
        self.client.close()
        self.wait_for({1: False, 4: False, 5: True, 6: False, 7: False})
        self.assertFalse(self.client._thread.is_alive())

    def test_heartbeat_resynchronizes_after_board_reboot(self):
        self.client.update("green")
        self.wait_for({1: True, 7: True})
        with self.condition:
            self.levels.clear()
        self.wait_for({1: True, 4: False, 5: False, 6: False, 7: True})

    def test_gate_releases_after_green_and_blocks_before_lamp_changes(self):
        self.client.update('green')
        self.wait_for({1: True, 7: True})
        with self.condition:
            commands = self.commands.copy()
            first_release = commands.index((1, 'on'))
            self.assertLess(commands.index((7, 'on')), first_release)
            offset = len(commands)
        self.client.update('red')
        self.wait_for({1: False, 6: True, 7: False})
        with self.condition:
            commands = self.commands[offset:]
            self.assertEqual(commands[0], (1, 'off'))
        self.client.update('orange')
        self.wait_for({1: False, 5: True, 6: False, 7: False})

    def test_gate_debounce_response_is_retried_until_confirmed(self):
        with self.condition:
            self.ignore_gate_commands = 2
        self.client.update('green')
        self.wait_for({1: True, 7: True})
        with self.condition:
            self.assertGreaterEqual(self.commands.count((1, 'on')), 3)

    def test_close_retries_gate_block_when_firmware_ignores_first_command(self):
        self.client.update('green')
        self.wait_for({1: True, 7: True})
        with self.condition:
            self.ignore_gate_commands = 1
        self.client.close()
        self.wait_for({1: False, 5: True, 7: False})
        self.assertFalse(self.client._thread.is_alive())

    def test_http_failure_does_not_block_updates_and_retries(self):
        with self.condition:
            self.fail = True
        started = time.monotonic()
        self.client.update("red")
        self.assertLess(time.monotonic() - started, .1)
        deadline = time.monotonic() + 3
        while self.client.status["last_error"] is None and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(self.client.status["connected"])
        self.assertIsNotNone(self.client.status["last_error"])
        with self.condition:
            self.fail = False
        self.wait_for({4: False, 5: False, 6: True, 7: False})


if __name__ == "__main__":
    unittest.main()

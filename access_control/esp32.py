from __future__ import annotations

import http.client
import json
import threading
import time
from urllib.parse import urlsplit


DEFAULT_ESP_URL = "http://192.168.3.1"
LAMP_PINS = {"orange": 5, "red": 6, "green": 7}
GATE_PIN = 1


class EspGpioClient:
    """Coalesce GPIO commands in a worker; inference never waits on HTTP."""

    def __init__(self, base_url: str, timeout: float = 0.7,
                 heartbeat: float = 5.0, retry_interval: float = 1.0):
        self.url = base_url.rstrip("/")
        self._address = urlsplit(self.url)
        if (self._address.scheme not in ("http", "https") or not self._address.hostname
                or self._address.path or self._address.query or self._address.fragment
                or self._address.username or self._address.password):
            raise ValueError("EspUrl deve ser http://IP:porta ou https://IP:porta")
        if min(timeout, heartbeat, retry_interval) <= 0:
            raise ValueError("Os intervalos HTTP devem ser positivos")
        self.timeout = timeout
        self.heartbeat = heartbeat
        self.retry_interval = retry_interval
        self._condition = threading.Condition()
        self._lamp = "orange"
        self._alarm_until = 0.0
        self._closing = False
        self._close_deadline = 0.0
        self._connected = False
        self._last_error: str | None = None
        self._thread = threading.Thread(target=self._run, name="esp32-gpio", daemon=True)
        self._thread.start()

    def update(self, lamp: str, alarm_until: float = 0.0):
        if lamp not in LAMP_PINS:
            raise ValueError(f"Sinal desconhecido: {lamp}")
        with self._condition:
            if not self._closing and (lamp, alarm_until) != (self._lamp, self._alarm_until):
                self._lamp, self._alarm_until = lamp, alarm_until
                self._condition.notify_all()

    @property
    def status(self) -> dict:
        with self._condition:
            return {"url": self.url, "connected": self._connected,
                    "last_error": self._last_error}

    def close(self):
        # Block the gate and return to orange, even after Ctrl+C.
        with self._condition:
            self._closing = True
            self._close_deadline = time.monotonic() + 5 * self.timeout + 1.0
            self._lamp, self._alarm_until = "orange", 0.0
            self._condition.notify_all()
        self._thread.join(timeout=5 * self.timeout + 1.5)

    def _levels(self, now: float) -> dict[int, bool]:
        return {GATE_PIN: self._lamp == "green", 4: now < self._alarm_until,
                **{pin: self._lamp == lamp for lamp, pin in LAMP_PINS.items()}}

    def _send(self, pin: int, enabled: bool):
        connection_type = (http.client.HTTPSConnection if self._address.scheme == "https"
                           else http.client.HTTPConnection)
        connection = connection_type(self._address.hostname, self._address.port, timeout=self.timeout)
        action = "on" if enabled else "off"
        try:
            if pin == GATE_PIN:
                connection.request("POST", "/api/state",
                                   body=json.dumps({"gate_action": action}),
                                   headers={"Connection": "close", "Content-Type": "application/json"})
            else:
                connection.request("GET", f"/gpio{pin}:{action}", headers={"Connection": "close"})
            response = connection.getresponse()
            payload = json.loads(response.read(65536))
            if response.status != 200 or not isinstance(payload, dict) or payload.get("status") != "ok":
                raise OSError(f"GPIO{pin}: resposta invalida (HTTP {response.status})")
            # The firmware may ignore a transition during its relay debounce.
            # Only acknowledge the command after it reports the requested state.
            if pin == GATE_PIN and (payload.get("gate_state") != action
                                    or payload.get("gate_enabled") is not True):
                raise OSError(f"Portao: estado solicitado {action} nao confirmado")
        finally:
            connection.close()

    def _run(self):
        sent: dict[int, bool] = {}
        next_refresh = 0.0
        warned = False
        while True:
            with self._condition:
                now = time.monotonic()
                desired = self._levels(now)
                if now >= next_refresh:
                    sent.clear()
                changes = [(pin, enabled) for pin, enabled in desired.items()
                           if sent.get(pin) != enabled]
                # Block first; release only after the green lamp is confirmed.
                changes.sort(key=lambda item: (item[1],
                             -1 if item == (GATE_PIN, False) else
                             1 if item == (GATE_PIN, True) else 0))
                if not changes:
                    if self._closing:
                        return
                    delay = max(0.01, next_refresh - now)
                    if self._alarm_until > now:
                        delay = min(delay, self._alarm_until - now)
                    self._condition.wait(timeout=delay)
                    continue
            try:
                stale = False
                for pin, enabled in changes:
                    with self._condition:
                        if self._levels(time.monotonic()) != desired:
                            stale = True
                            break
                    self._send(pin, enabled)
                    sent[pin] = enabled
                if stale:
                    continue
                with self._condition:
                    self._connected = True
                    self._last_error = None
                    next_refresh = time.monotonic() + self.heartbeat
                if warned:
                    print("ESP32 reconectado; sinalizacao sincronizada.", flush=True)
                    warned = False
            except (OSError, ValueError, http.client.HTTPException) as error:
                sent.clear()
                with self._condition:
                    self._connected = False
                    self._last_error = str(error)
                    if not warned:
                        print(f"ESP32 indisponivel em {self.url}: {error}. Tentando novamente...",
                              flush=True)
                        warned = True
                    if self._closing and time.monotonic() >= self._close_deadline:
                        return
                    self._condition.wait(timeout=min(self.retry_interval, .1) if self._closing
                                         else self.retry_interval)

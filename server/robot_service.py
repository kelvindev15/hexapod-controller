#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.utils.logging_config import bootstrap_logging
from common.utils.images import toBase64Image
from server.core.api_contract import ContractValidationError, action_from_dict, action_to_dict, world_state_to_dict
from server.core.motion_schema import Action, ActionType
from server.drivers.camera import Camera


logger = logging.getLogger(__name__)


class RobotHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, request_handler_class, executor, camera=None):
        super().__init__(server_address, request_handler_class)
        self.executor = executor
        self.camera = camera


def create_handler():
    class RobotServiceHandler(BaseHTTPRequestHandler):
        server_version = "HexapodRobotService/1.0"

        def log_message(self, format, *args):
            logger.info("robot-service %s - %s", self.address_string(), format % args)

        def _read_json(self):
            length_header = self.headers.get("Content-Length", "0")
            try:
                content_length = int(length_header)
            except Exception as exc:
                raise ContractValidationError("Invalid Content-Length") from exc

            if content_length <= 0:
                return {}

            raw = self.rfile.read(content_length)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception as exc:
                raise ContractValidationError("Request body must be valid JSON") from exc

        def _send_json(self, status_code: int, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "running": bool(getattr(self.server.executor, "running", True)),
                    },
                )
                return

            if self.path == "/state":
                state = self.server.executor.get_state()
                self._send_json(200, world_state_to_dict(state))
                return

            if self.path == "/camera":
                try:
                    if not self.server.camera:
                        self._send_json(503, {"error": "Camera not available"})
                        return
                    image = self.server.camera.capture_array()
                    if image is not None:
                        base64_image = toBase64Image(image)
                        self._send_json(200, {"image": base64_image})
                        return
                    self._send_json(500, {"error": "Failed to capture camera image"})
                except Exception as exc:
                    logger.exception("Failed to get camera image")
                    self._send_json(500, {"error": str(exc)})
                return

            self._send_json(404, {"error": "Not found"})

        def do_POST(self):
            if self.path == "/actions":
                try:
                    payload = self._read_json()
                    action = action_from_dict(payload)
                    self.server.executor.submit_action(action)
                    self._send_json(202, {"accepted": True, "action": action_to_dict(action)})
                except ContractValidationError as exc:
                    self._send_json(400, {"error": str(exc)})
                except Exception as exc:
                    logger.exception("Failed to submit action")
                    self._send_json(500, {"error": f"Failed to submit action: {exc}"})
                return

            if self.path == "/actions/stop":
                try:
                    action = Action(type=ActionType.STOP)
                    self.server.executor.submit_action(action)
                    self._send_json(202, {"accepted": True, "action": action_to_dict(action)})
                except Exception as exc:
                    logger.exception("Failed to submit stop action")
                    self._send_json(500, {"error": f"Failed to submit stop action: {exc}"})
                return

            self._send_json(404, {"error": "Not found"})

    return RobotServiceHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robot-side HTTP motion service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    return parser


def main() -> int:
    bootstrap_logging()
    args = build_parser().parse_args()

    from server.core.motion_executor import MotionExecutor

    executor = MotionExecutor()
    executor.start()
    
    camera = None
    try:
        camera = Camera()
        logger.info("Camera initialized successfully")
    except Exception as exc:
        logger.warning("Camera initialization failed: %s", exc)
    
    server = RobotHTTPServer((args.host, args.port), create_handler(), executor, camera)
    logger.info("Robot service listening on %s:%d", args.host, args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Robot service interrupted")
    finally:
        server.server_close()
        executor.stop()
        if camera:
            try:
                camera.close()
            except Exception as exc:
                logger.warning("Error closing camera: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
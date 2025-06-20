# (백그라운드 스레드에서 실행)
import jwt
import websocket, json, hmac, hashlib, time, uuid, os


def _jwt_token():
    payload = {"access_key": os.getenv("UPBIT_ACCESS_KEY"), "nonce": str(uuid.uuid4())}
    jwt_token = jwt.encode(payload, os.getenv("UPBIT_SECRET_KEY"))
    return jwt_token


def start_myasset_stream(callback):
    url = "wss://api.upbit.com/websocket/v1/private"

    def on_open(ws):
        auth = {
            "ticket": str(uuid.uuid4()),
            "type": "authenticated",
            "access_token": _jwt_token(),
        }
        ws.send(
            json.dumps(
                [
                    auth,
                    {"type": "myAsset"},
                ]
            )
        )

    def on_message(ws, msg):
        data = json.loads(msg)
        if data["type"] == "myAsset":
            callback(data)  # 실시간 잔고 업데이트

    websocket.WebSocketApp(url, on_open=on_open, on_message=on_message).run_forever()

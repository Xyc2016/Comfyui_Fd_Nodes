import json
from datetime import datetime

import requests


def webhook_send(url: str, content: dict):
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    headers = {"Content-Type": "application/json"}
    data = {"msgtype": "text", "text": {"content": json.dumps({"webhook_sent": now_str, **content}, ensure_ascii=False, indent=4)}}

    requests.post(url, headers=headers, json=data)
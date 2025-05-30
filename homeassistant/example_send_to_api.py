import requests

HASS_URL = "http://homeassistant.local:8123"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiIzZDFhODJjOTU4NTc0ZTI3OTkxZmI5ZjYzNDI4MGJiYiIsImlhdCI6MTc0ODUxOTc5NywiZXhwIjoyMDYzODc5Nzk3fQ.wMGCZgf4qfURtRjaAMgbzOcSzBjSJaIG3a46P8_DRjY"

url = f"{HASS_URL}/api/tank1-microvolts"

payload = {
    "HwUid": "sensor-node-123",
    "AboutNodeNameList": ["tank1_depth1", "tank1_depth2"],
    "MicroVoltsList": [0.7*1e6, 0.8*1e6],
    "TypeName": "microvolts",
    "Version": "100"
}

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, json=payload)

print(f"Status Code: {response.status_code}")
print("Response:")
print(response.json())


# # Simple example
# import requests
# url = "http://192.168.1.112:8123/api/my_api"
# headers = {
#     "Authorization": f"Bearer {token}",
#     "Content-Type": "application/json",
# }
# data = {
#     "sensor_id": "tank1_depth1",
#     "temperature": 12
# }
# response = requests.post(url, json=data, headers=headers)
# try:
#     json_data = response.json()
#     print("Response JSON:", json_data)
# except requests.exceptions.JSONDecodeError:
#     print("Response is not valid JSON.")

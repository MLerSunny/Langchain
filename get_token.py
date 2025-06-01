import requests
import json

# API endpoint
url = "http://localhost:8000/auth/token"

# Request payload
payload = {
    "email": "test@example.com",
    "password": "testpassword123",
    "username": "testuser",
    "full_name": "Test User"
}

# Headers
headers = {
    "Content-Type": "application/json"
}

# Make the request
response = requests.post(url, json=payload, headers=headers)

# Print response
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}") 
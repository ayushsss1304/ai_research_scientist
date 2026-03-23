import requests

api_key = '3228s62pqbzh2sx276rbu5dp'

# Test the API
url = 'http://ieeexploreapi.ieee.org/api/v1/search/articles'
params = {
    'apikey': api_key,
    'querytext': 'machine learning',
    'max_records': 1
}

response = requests.get(url, params=params)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.text[:500]}")

if response.status_code == 200:
    print("\n✅ IEEE API key is VALID!")
    data = response.json()
    print(f"Total records: {data.get('total_records', 0)}")
elif response.status_code == 403:
    print("\n❌ IEEE API key is INVALID or EXPIRED")
    print("Get a new key at: https://developer.ieee.org/")
elif response.status_code == 401:
    print("\n❌ IEEE API authentication failed")
else:
    print(f"\n⚠️ Unexpected status: {response.status_code}")
import os
import time
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

ASIAN_CITIES = [
    # Singapore
    {"name": "Singapore", "district": "Marina Bay",       "lat": 1.2834,  "lng": 103.8607},
    {"name": "Singapore", "district": "Chinatown",        "lat": 1.2836,  "lng": 103.8448},
    {"name": "Singapore", "district": "Little India",     "lat": 1.3066,  "lng": 103.8518},
    {"name": "Singapore", "district": "Orchard",          "lat": 1.3048,  "lng": 103.8318},
    {"name": "Singapore", "district": "Kampong Glam",     "lat": 1.3019,  "lng": 103.8590},

    # Bangkok
    {"name": "Bangkok",   "district": "Sukhumvit",        "lat": 13.7317, "lng": 100.5670},
    {"name": "Bangkok",   "district": "Silom",            "lat": 13.7250, "lng": 100.5287},
    {"name": "Bangkok",   "district": "Old Town",         "lat": 13.7533, "lng": 100.4934},
    {"name": "Bangkok",   "district": "Chatuchak",        "lat": 13.7999, "lng": 100.5500},
    {"name": "Bangkok",   "district": "Thonglor",         "lat": 13.7278, "lng": 100.5836},

    # Tokyo
    {"name": "Tokyo",     "district": "Shinjuku",         "lat": 35.6938, "lng": 139.7034},
    {"name": "Tokyo",     "district": "Shibuya",          "lat": 35.6580, "lng": 139.7016},
    {"name": "Tokyo",     "district": "Asakusa",          "lat": 35.7148, "lng": 139.7967},
    {"name": "Tokyo",     "district": "Harajuku",         "lat": 35.6702, "lng": 139.7026},
    {"name": "Tokyo",     "district": "Ginza",            "lat": 35.6717, "lng": 139.7649},
    {"name": "Tokyo",     "district": "Shimokitazawa",    "lat": 35.6614, "lng": 139.6680},

    # Seoul
    {"name": "Seoul",     "district": "Gangnam",          "lat": 37.4979, "lng": 127.0276},
    {"name": "Seoul",     "district": "Insadong",         "lat": 37.5744, "lng": 126.9858},
    {"name": "Seoul",     "district": "Hongdae",          "lat": 37.5563, "lng": 126.9234},
    {"name": "Seoul",     "district": "Itaewon",          "lat": 37.5345, "lng": 126.9940},
    {"name": "Seoul",     "district": "Bukchon",          "lat": 37.5822, "lng": 126.9833},

    # Kuala Lumpur
    {"name": "Kuala Lumpur", "district": "KLCC",          "lat": 3.1579,  "lng": 101.7117},
    {"name": "Kuala Lumpur", "district": "Bukit Bintang", "lat": 3.1478,  "lng": 101.7107},
    {"name": "Kuala Lumpur", "district": "Bangsar",       "lat": 3.1287,  "lng": 101.6791},
    {"name": "Kuala Lumpur", "district": "Chow Kit",      "lat": 3.1706,  "lng": 101.6988},

    # Jakarta
    {"name": "Jakarta",   "district": "Kemang",           "lat": -6.2607, "lng": 106.8148},
    {"name": "Jakarta",   "district": "SCBD",             "lat": -6.2241, "lng": 106.8095},
    {"name": "Jakarta",   "district": "Menteng",          "lat": -6.1963, "lng": 106.8317},
    {"name": "Jakarta",   "district": "Kelapa Gading",    "lat": -6.1580, "lng": 106.9042},

    # Manila
    {"name": "Manila",    "district": "BGC",              "lat": 14.5505, "lng": 121.0495},
    {"name": "Manila",    "district": "Makati",           "lat": 14.5547, "lng": 121.0244},
    {"name": "Manila",    "district": "Intramuros",       "lat": 14.5890, "lng": 120.9750},
    {"name": "Manila",    "district": "Poblacion",        "lat": 14.5651, "lng": 121.0297},

    # Shanghai
    {"name": "Shanghai",  "district": "The Bund",         "lat": 31.2365, "lng": 121.4905},
    {"name": "Shanghai",  "district": "French Concession","lat": 31.2197, "lng": 121.4553},
    {"name": "Shanghai",  "district": "Xintiandi",        "lat": 31.2193, "lng": 121.4730},
    {"name": "Shanghai",  "district": "Jing'an",          "lat": 31.2282, "lng": 121.4474},
    {"name": "Shanghai",  "district": "Tianzifang",       "lat": 31.2098, "lng": 121.4669},

    # Beijing
    {"name": "Beijing",   "district": "Sanlitun",         "lat": 39.9330, "lng": 116.4551},
    {"name": "Beijing",   "district": "Hutong",           "lat": 39.9390, "lng": 116.3980},
    {"name": "Beijing",   "district": "Wudaokou",         "lat": 39.9921, "lng": 116.3128},
    {"name": "Beijing",   "district": "Guomao",           "lat": 39.9076, "lng": 116.4607},

    # Taipei
    {"name": "Taipei",    "district": "Daan",             "lat": 25.0267, "lng": 121.5432},
    {"name": "Taipei",    "district": "Ximending",        "lat": 25.0424, "lng": 121.5078},
    {"name": "Taipei",    "district": "Zhongshan",        "lat": 25.0630, "lng": 121.5239},
    {"name": "Taipei",    "district": "Xinyi",            "lat": 25.0330, "lng": 121.5654},

    # Hong Kong
    {"name": "Hong Kong", "district": "Central",          "lat": 22.2820, "lng": 114.1588},
    {"name": "Hong Kong", "district": "Wan Chai",         "lat": 22.2769, "lng": 114.1731},
    {"name": "Hong Kong", "district": "Mong Kok",         "lat": 22.3193, "lng": 114.1694},
    {"name": "Hong Kong", "district": "Causeway Bay",     "lat": 22.2802, "lng": 114.1837},
    {"name": "Hong Kong", "district": "Sham Shui Po",     "lat": 22.3308, "lng": 114.1625},

    # Osaka
    {"name": "Osaka",     "district": "Dotonbori",        "lat": 34.6687, "lng": 135.5013},
    {"name": "Osaka",     "district": "Namba",            "lat": 34.6659, "lng": 135.5015},
    {"name": "Osaka",     "district": "Umeda",            "lat": 34.7024, "lng": 135.4959},
    {"name": "Osaka",     "district": "Shinsekai",        "lat": 34.6518, "lng": 135.5063},
]

RAW_DATA_DIR = Path("data/raw")
PHOTOS_DIR   = RAW_DATA_DIR / "photos"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


def search_restaurants(lat: float, lng: float, city: str, district: str, max_results: int = 20) -> list[dict]:
    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.formattedAddress,"
            "places.rating,places.userRatingCount,places.photos,"
            "places.types,places.location"
        ),
    }
    body = {
        "includedTypes": ["restaurant", "cafe"],
        "maxResultCount": max_results,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": 1500.0,
            }
        },
        "rankPreference": "POPULARITY",
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        print(f"[ERROR] {city}/{district}: {response.status_code} - {response.text[:200]}")
        return []
    return response.json().get("places", [])


def download_photo(photo_name: str, city: str, place_id: str, idx: int) -> str | None:
    url = f"https://places.googleapis.com/v1/{photo_name}/media"
    params = {"maxHeightPx": 600, "maxWidthPx": 600, "key": API_KEY}
    response = requests.get(url, params=params, timeout=15)
    if response.status_code != 200:
        return None
    filename = f"{city.replace(' ','_')}_{place_id[:12]}_{idx}.jpg"
    filepath = PHOTOS_DIR / filename
    with open(filepath, "wb") as f:
        f.write(response.content)
    return str(filepath)


def collect_all_cities(max_per_district: int = 20) -> list[dict]:
    all_records = []
    seen_place_ids = set()

    for location in tqdm(ASIAN_CITIES, desc="Districts"):
        city     = location["name"]
        district = location["district"]

        places = search_restaurants(
            location["lat"], location["lng"],
            city, district,
            max_results=max_per_district,
        )

        for place in places:
            place_id = place.get("id", "")
            if place_id in seen_place_ids:
                continue
            seen_place_ids.add(place_id)

            name        = place.get("displayName", {}).get("text", "Unknown")
            address     = place.get("formattedAddress", "")
            rating      = place.get("rating", 0)
            num_ratings = place.get("userRatingCount", 0)
            location_ll = place.get("location", {})
            photos_meta = place.get("photos", [])

            photo_paths = []
            for i, photo in enumerate(photos_meta[:3]):
                photo_name = photo.get("name", "")
                if photo_name:
                    path = download_photo(photo_name, city, place_id, i)
                    if path:
                        photo_paths.append(path)
                time.sleep(0.08)

            if not photo_paths:
                continue

            all_records.append({
                "place_id":    place_id,
                "name":        name,
                "city":        city,
                "district":    district,
                "country":     _city_to_country(city),
                "address":     address,
                "rating":      rating,
                "num_ratings": num_ratings,
                "lat":         location_ll.get("latitude"),
                "lng":         location_ll.get("longitude"),
                "photo_paths": photo_paths,
            })

        time.sleep(0.3)

    out_path = RAW_DATA_DIR / "restaurants.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Collected {len(all_records)} restaurants across {len(ASIAN_CITIES)} districts → {out_path}")
    return all_records


def _city_to_country(city: str) -> str:
    mapping = {
        "Singapore":    "Singapore",
        "Bangkok":      "Thailand",
        "Tokyo":        "Japan",
        "Osaka":        "Japan",
        "Seoul":        "South Korea",
        "Kuala Lumpur": "Malaysia",
        "Jakarta":      "Indonesia",
        "Manila":       "Philippines",
        "Shanghai":     "China",
        "Beijing":      "China",
        "Taipei":       "Taiwan",
        "Hong Kong":    "Hong Kong",
    }
    return mapping.get(city, "Asia")


if __name__ == "__main__":
    records = collect_all_cities(max_per_district=20)
    print(f"Total restaurants with photos: {len(records)}")
# Easysort Customer API — Client Guide

This API lets your organisation pull your recycling results programmatically. You get a
single **API key**; every request must include it. You can only ever see data that belongs
to your organisation.

- **Base URL:** `https://<the-host-we-give-you>` (e.g. `https://api.easysort.org`)
- **Auth:** `Authorization: Bearer <YOUR_API_KEY>` header on every request
- **Format:** JSON over HTTPS
- **API version:** `v1` (the version is part of the path, e.g. `/v1/results`)

> Keep your API key secret. Treat it like a password — anyone with the key can read your
> data. If it leaks, tell us and we will rotate it. We never send the key over email in
> plain text more than once.

---

## Endpoints

### 1. Health check — `GET /v1/health`

No authentication required. Use it to confirm the service is reachable.

```bash
curl https://api.easysort.org/v1/health
```

```json
{ "status": "ok" }
```

### 2. List available periods — `GET /v1/results`

Returns the period identifiers you can fetch. Periods are weekly (and monthly) buckets.

```bash
curl https://api.easysort.org/v1/results \
  -H "Authorization: Bearer $EASYSORT_API_KEY"
```

```json
{
  "organisation": "ARGO",
  "results": ["week_21_2026", "week_22_2026", "week_23_2026"]
}
```

- `week_<week>_<year>` — an ISO-week period (Monday–Sunday).
- `month_<month>_<year>` — a calendar-month period (if enabled for you).

### 3. List your locations — `GET /v1/locations`

Returns the drop-off locations seen in your **most recent 6 weekly results**, so you know which
names to expect in the period results (the location names are the per-location keys inside each
result). Using a recent window keeps the list current with where you're actively reporting.

```bash
curl https://api.easysort.org/v1/locations \
  -H "Authorization: Bearer $EASYSORT_API_KEY"
```

```json
{
  "organisation": "ARGO",
  "locations": ["Jyllinge", "Roskilde"]
}
```

### 4. Get one period — `GET /v1/results/{period}`

Pass one of the identifiers returned above.

```bash
curl https://api.easysort.org/v1/results/week_23_2026 \
  -H "Authorization: Bearer $EASYSORT_API_KEY"
```

```json
{
  "date_start": "01_06_2026",
  "date_end": "07_06_2026",
  "Hørgården": {
    "objects": "412",
    "weight_kg": "63",
    "visitors": "180",
    "categories": [
      { "category": "Plastic", "count": "120", "weight_kg": "18" },
      { "category": "Glass", "count": "64", "weight_kg": "22" }
    ],
    "objects_per_day": [
      { "day": "Monday", "count": "58" },
      { "day": "Tuesday", "count": "61" }
    ],
    "objects_per_hour": [
      { "hour": "0-3", "count": "2" },
      { "hour": "9-12", "count": "140" }
    ]
  }
}
```

#### Field reference

| Field | Meaning |
|---|---|
| `date_start` / `date_end` | Period boundaries, formatted `DD_MM_YYYY`. |
| `<location>` | One object per drop-off location in your deployment. |
| `objects` | Total registered items in the period. |
| `weight_kg` | Estimated total weight, in kilograms. |
| `visitors` | Estimated number of visitors. |
| `categories[]` | Per-material breakdown: `category`, `count`, `weight_kg`. |
| `objects_per_day[]` | Items per weekday (`Monday`…`Sunday`). |
| `objects_per_hour[]` | Items per 3-hour bucket (`"0-3"`, `"3-6"`, … `"21-24"`). |

> All numeric values are returned as **strings** containing rounded integers (e.g. `"63"`).
> Parse them with `int(...)` / `parseInt(...)` on your side.

---

## Example: Python

```python
import requests

BASE = "https://api.easysort.org"
KEY = "YOUR_API_KEY"  # load from an env var / secret manager in real code
headers = {"Authorization": f"Bearer {KEY}"}

periods = requests.get(f"{BASE}/v1/results", headers=headers, timeout=30).json()
latest = periods["results"][-1]

data = requests.get(f"{BASE}/v1/results/{latest}", headers=headers, timeout=30).json()
for location, summary in data.items():
    if location in ("date_start", "date_end"):
        continue
    print(location, "->", summary["objects"], "items,", summary["weight_kg"], "kg")
```

## Example: JavaScript (Node 18+)

```js
const BASE = "https://api.easysort.org";
const KEY = process.env.EASYSORT_API_KEY;
const headers = { Authorization: `Bearer ${KEY}` };

const periods = await (await fetch(`${BASE}/v1/results`, { headers })).json();
const latest = periods.results.at(-1);
const data = await (await fetch(`${BASE}/v1/results/${latest}`, { headers })).json();
console.log(data);
```

---

## Errors

The API uses standard HTTP status codes:

| Status | Meaning | What to do |
|---|---|---|
| `200` | Success | — |
| `400` | Malformed `period` identifier | Use an identifier returned by `GET /v1/results`. |
| `401` | Missing / invalid API key | Check the `Authorization: Bearer <key>` header. |
| `404` | That period does not exist for you | List periods first; it may not be published yet. |
| `503` | Backend storage temporarily unavailable | Retry with backoff. |

Error bodies look like:

```json
{ "detail": "Invalid API key." }
```

---

## Notes & guarantees

- You only ever see **your** organisation's data; the key determines the scope.
- New periods appear automatically once we publish them — poll `GET /v1/results`
  (e.g. once a day) to discover them.
- Please cache results on your side; data for a closed period does not change.
- Rate-limit yourself to something reasonable (a handful of requests per minute is plenty).

Questions or a key rotation? Contact Easysort.

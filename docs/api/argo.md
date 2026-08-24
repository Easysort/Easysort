# Easysort Customer API — Client Guide

This API lets your organisation pull your recycling results programmatically (and, if
enabled for your key, Vision+ device videos). You get a single **API key**; every request
must include it. Recycling results are scoped to your organisation; Vision+ list/download
(when granted) covers the shared Vision+ video bucket.

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
curl -H "Authorization: Bearer $EASYSORT_API_KEY" https://api.easysort.org/v1/results
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
curl -H "Authorization: Bearer $EASYSORT_API_KEY" https://api.easysort.org/v1/locations
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
curl -H "Authorization: Bearer $EASYSORT_API_KEY" https://api.easysort.org/v1/results/week_23_2026
```

```json
{
  "date_start": "01_06_2026",
  "date_end": "07_06_2026",
  "Hørgården": {
    "objects": "412",
    "weight_kg": "63",
    "co2_kg": "88",
    "co2_kg_low": "62",
    "co2_kg_high": "106",
    "visitors": "180",
    "percentage_personnel": "12",
    "percentage_citizens": "88",
    "categories": [
      {
        "category": "Plastic",
        "count": "120",
        "weight_kg": "18",
        "objects_per_hour": [
          { "hour": "0-3", "count": "0" },
          { "hour": "9-12", "count": "74" }
        ]
      },
      { "category": "Glass", "count": "64", "weight_kg": "22", "objects_per_hour": [] }
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
| `co2_kg` | Estimated CO₂ saved, in kilograms (best estimate). |
| `co2_kg_low` / `co2_kg_high` | Lower/upper bound of the CO₂ estimate. |
| `visitors` | Estimated number of visitors. |
| `percentage_personnel` | Share of `objects` registered as personnel activity, as a percentage. |
| `percentage_citizens` | Share of `objects` registered as citizen activity, as a percentage. Together with `percentage_personnel` this sums to 100. |
| `categories[]` | Per-material breakdown: `category`, `count`, `weight_kg`, `objects_per_hour`. |
| `categories[].objects_per_hour[]` | That material's items per 3-hour bucket. Same bucket labels as the location-level series, and sums to the category's `count`. |
| `objects_per_day[]` | Items per weekday (`Monday`…`Sunday`). |
| `objects_per_hour[]` | Items per 3-hour bucket (`"0-3"`, `"3-6"`, … `"21-24"`). |
> Where the per-material breakdown is present, summing `objects_per_hour` across all entries of
> `categories[]` reproduces the location-level `objects_per_hour`, and summing a category's
> `objects_per_hour` reproduces its `count`. So you can slice the hourly flow either by material or
> by location without reconciling two different totals.

**Availability of `categories[].objects_per_hour`:** this breakdown starts with **week 32 of 2026**
for weekly periods and with **September 2026** for monthly periods; earlier periods return an empty
array (`[]`), so use the weekly periods if you need hourly material detail before September. A
location may also return an empty array for a period it was not broken down for, so treat `[]` as
"not available here" rather than as zeroes, and check the array before you read it. The
location-level `objects_per_hour` covers your full history in every period type.

> All numeric values are returned as **strings** containing rounded integers (e.g. `"63"`).
> Parse them with `int(...)` / `parseInt(...)` on your side.

### 5. Daily results — `GET /v1/days` and `GET /v1/results/day_DD_MM_YYYY`

If you want per-day numbers instead of whole weeks, fetch a single day. The identifier is
`day_DD_MM_YYYY` (e.g. `day_03_06_2026`). You can build it directly from any date, or list the
days that are available:

```bash
# List available day identifiers (oldest first):
curl -H "Authorization: Bearer $EASYSORT_API_KEY" https://api.easysort.org/v1/days

# Fetch one day:
curl -H "Authorization: Bearer $EASYSORT_API_KEY" https://api.easysort.org/v1/results/day_03_06_2026
```

```json
{
  "organisation": "ARGO",
  "days": ["day_01_06_2026", "day_02_06_2026", "day_03_06_2026"]
}
```

A day response has the **same per-location fields** as a period (`objects`, `weight_kg`,
`co2_kg`, …, `categories[]`), with two differences:

- `date_start` and `date_end` are both that single day.
- `objects_per_day` collapses to a single entry — the weekday that day falls on.

```json
{
  "date_start": "03_06_2026",
  "date_end": "03_06_2026",
  "Roskilde": {
    "objects": "142",
    "weight_kg": "201",
    "co2_kg": "30",
    "co2_kg_low": "21",
    "co2_kg_high": "36",
    "visitors": "58",
    "percentage_personnel": "9",
    "percentage_citizens": "91",
    "categories": [
      {
        "category": "Møbler og indretning",
        "count": "142",
        "weight_kg": "201",
        "objects_per_hour": [
          { "hour": "9-12", "count": "61" },
          { "hour": "12-15", "count": "70" }
        ]
      }
    ],
    "objects_per_day": [{ "day": "Wednesday", "count": "142" }],
    "objects_per_hour": [
      { "hour": "9-12", "count": "61" },
      { "hour": "12-15", "count": "70" }
    ]
  }
}
```

> Summing a location's seven days in a week reproduces that week's total, so you can compute **any**
> total you like yourself — a day, a custom date range, or all locations combined — by fetching the
> days you need and adding them. `GET /v1/results` only lists weeks/months; use `GET /v1/days` (or
> build `day_DD_MM_YYYY`) for days.

---

## Ready-made CSV converters

If you would rather work in Excel or Power BI than in JSON, two scripts ship alongside this
document. Both take a file you downloaded from the API and write CSVs next to it. They need only
Python 3.9+ and the standard library — no packages to install.

```bash
# A week or a month:
python week_to_csv.py week_34_2026.json
python week_to_csv.py month_8_2026.json --out-dir ./reports

# A single day:
python day_to_csv.py day_20_08_2026.json
```

Each run writes four files:

| File | Contents |
|---|---|
| `<name>_objects.csv` | One row per location: objects, weight, CO₂, visitors. |
| `<name>_totals.csv` | Organisation-wide totals for the period. |
| `<name>_per_day.csv` | Objects per weekday, one column per location (weeks/months). |
| `<name>_per_hour.csv` | Objects per 3-hour bucket, per location **and per category** — long format, ready to pivot. Where a location has no per-material breakdown for the period, its hourly flow appears under the category `All categories` instead, so the hourly view is filled in for every period. |

For a day the third file is `<name>_categories.csv` (the per-material breakdown) instead of
`_per_day.csv`, since a day has only one weekday.

## Worked example: from raw response to the numbers you want

The typical flow is always the same three steps:

1. `GET /v1/results` → pick the period you want (e.g. the latest week).
2. `GET /v1/results/{period}` → get one JSON object. Its top-level keys are
   `date_start`, `date_end`, and **one key per location**.
3. Loop over the location keys, skipping `date_start` / `date_end`, and read the
   fields you need. Remember every number is a **string** — wrap it in `int(...)`.

Given a period response like:

```json
{
  "date_start": "01_06_2026",
  "date_end": "07_06_2026",
  "Jyllinge": { "objects": "412", "weight_kg": "63", "co2_kg": "88", "visitors": "180", "categories": [ ... ] },
  "Roskilde": { "objects": "988", "weight_kg": "141", "co2_kg": "205", "visitors": "402", "categories": [ ... ] }
}
```

...here is how to derive the three things people usually ask for.

### Python

```python
import requests

BASE = "https://api.easysort.org"
KEY = "YOUR_API_KEY"  # load from an env var / secret manager in real code
headers = {"Authorization": f"Bearer {KEY}"}

# 1 + 2: pick the latest period and download it.
periods = requests.get(f"{BASE}/v1/results", headers=headers, timeout=30).json()
latest = periods["results"][-1]
data = requests.get(f"{BASE}/v1/results/{latest}", headers=headers, timeout=30).json()

# The location keys are everything except the two date fields.
META = {"date_start", "date_end"}
locations = {name: summary for name, summary in data.items() if name not in META}

# a) Objects per location
objects_per_location = {name: int(summary["objects"]) for name, summary in locations.items()}
# -> {"Jyllinge": 412, "Roskilde": 988}

# b) Total objects across all locations
total_objects = sum(objects_per_location.values())
# -> 1400

# c) CO₂ per location (same pattern; use "co2_kg"). Weight works identically via "weight_kg".
co2_per_location = {name: int(summary["co2_kg"]) for name, summary in locations.items()}
# -> {"Jyllinge": 88, "Roskilde": 205}
total_co2 = sum(co2_per_location.values())

print(f"Period {data['date_start']}–{data['date_end']}")
for name in sorted(locations):
    print(f"  {name}: {objects_per_location[name]} objects, {co2_per_location[name]} kg CO₂")
print(f"  TOTAL: {total_objects} objects, {total_co2} kg CO₂")
```

### JavaScript (Node 18+)

```js
const BASE = "https://api.easysort.org";
const KEY = process.env.EASYSORT_API_KEY;
const headers = { Authorization: `Bearer ${KEY}` };

const periods = await (await fetch(`${BASE}/v1/results`, { headers })).json();
const latest = periods.results.at(-1);
const data = await (await fetch(`${BASE}/v1/results/${latest}`, { headers })).json();

const META = new Set(["date_start", "date_end"]);
const locations = Object.entries(data).filter(([name]) => !META.has(name));

// a) Objects per location
const objectsPerLocation = Object.fromEntries(
  locations.map(([name, s]) => [name, parseInt(s.objects, 10)]),
);

// b) Total objects across all locations
const totalObjects = Object.values(objectsPerLocation).reduce((a, b) => a + b, 0);

console.log(objectsPerLocation, "total:", totalObjects);
```

> **CO₂:** `co2_kg` is our best estimate of the CO₂ (in kg) saved by reusing the items
> registered at that location. `co2_kg_low` / `co2_kg_high` give a conservative lower/upper
> bound around it. Sum `co2_kg` across locations for an organisation-wide total, exactly like
> objects above.

---

### 6. Vision+ videos

If your organisation has Vision+ enabled, see [`VISION_PLUS_CUSTOMER.md`](./VISION_PLUS_CUSTOMER.md)
for device upload (signed URL) and list/download.

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

### `curl: (3) URL rejected: Bad hostname` (or "Malformed input to a URL")

This is **not** an API error — you'll usually see it *after* the correct JSON has already
printed. It means your shell handed `curl` an extra, garbled argument that it tried to open
as a second URL. It almost always comes from copy-pasting a multi-line command: a trailing
`\`, a hidden non-breaking space, or a Windows line-ending gets pulled in with the text.

Fixes:

- Use the **single-line** form of the commands in this guide (they no longer use `\`).
- Or type the command by hand instead of pasting.
- Your data is fine — the response you received above the error is complete.

---

## Notes & guarantees

- You only ever see **your** organisation's data; the key determines the scope.
- New periods appear automatically once we publish them — poll `GET /v1/results`
  (e.g. once a day) to discover them.
- Please cache results on your side; data for a closed period does not change.
- Rate-limit yourself to something reasonable (a handful of requests per minute is plenty).

Questions or a key rotation? Contact Easysort.

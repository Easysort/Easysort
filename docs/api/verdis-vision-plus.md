# Easysort Vision+ API — Customer Guide

This guide covers how to:

1. Upload MP4 videos from a device (device token → signed upload URL)
2. List and download those videos (API key → signed download URL)

- Base URL: `https://api.easysort.org` (confirm with Easysort if different)
- You receive two secrets from Easysort:
  - Device token — used to request an upload URL
  - API key — used to list videos and request download URLs

Signed URLs are valid for 5 minutes. Complete the upload/download within that window.

Keep both secrets confidential. If either leaks, tell Easysort and we will rotate it.

---

## 1. Health check (optional)

```bash
curl https://api.easysort.org/v1/health
```

```json
{ "status": "ok" }
```

---

## 2. Upload a video from a device

Uploads go in two steps: ask the API for a signed URL, then PUT the file directly to storage
(not through the API host). That keeps large transfers fast and reliable.

### Step A — get a signed upload URL

```bash
export DEVICE_TOKEN='<YOUR_DEVICE_TOKEN>'
export API_BASE='https://api.easysort.org'

curl -X POST \
  -H "Authorization: Bearer $DEVICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"filename":"2026-07-20T12-30-00Z.mp4"}' \
  "$API_BASE/v1/vision/upload-url"
```

`filename` is optional. If omitted, a UTC timestamp name is used (e.g. `2026-07-20T12-30-00Z.mp4`).

Example response:

```json
{
  "device": "prevas-test-01",
  "path": "prevas-test-01/2026-07-20T12-30-00Z.mp4",
  "upload_url": "https://….supabase.co/storage/v1/object/upload/sign/…?token=…",
  "token": "…",
  "method": "PUT",
  "expires_in": 300,
  "headers": { "Content-Type": "video/mp4" }
}
```

### Step B — PUT the file to `upload_url`

```bash
curl -X PUT \
  -H "Content-Type: video/mp4" \
  --data-binary @clip.mp4 \
  "$UPLOAD_URL"
```

Replace `$UPLOAD_URL` with the `upload_url` from step A (do not send the device token to Supabase).

### Upload rules

- File should be an MP4
- Use a simple filename (letters, digits, `.`, `_`, `-`) ending in `.mp4`
- Finish the PUT within `expires_in` seconds (300 = 5 minutes)
- Uploading again to the same path overwrites the previous file
- Need another device name/token? Ask Easysort

### Typical device integration

```bash
export DEVICE_TOKEN='…'
export API_BASE='https://api.easysort.org'
FILE=/path/to/recording.mp4
NAME="$(date -u +%Y-%m-%dT%H-%M-%SZ).mp4"

RESP="$(curl -sS -X POST \
  -H "Authorization: Bearer $DEVICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"filename\":\"$NAME\"}" \
  "$API_BASE/v1/vision/upload-url")"

UPLOAD_URL="$(printf '%s' "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["upload_url"])')"

curl -sS -X PUT \
  -H "Content-Type: video/mp4" \
  --data-binary @"$FILE" \
  "$UPLOAD_URL"
```

---

## 3. List videos

Use your API key (not the device token):

```bash
export EASYSORT_API_KEY='<YOUR_API_KEY>'
export API_BASE='https://api.easysort.org'

curl -H "Authorization: Bearer $EASYSORT_API_KEY" \
  "$API_BASE/v1/vision/videos"
```

Optional: only one device:

```bash
curl -H "Authorization: Bearer $EASYSORT_API_KEY" \
  "$API_BASE/v1/vision/videos?device=prevas-test-01"
```

Example response:

```json
{
  "organisation": "PREVAS",
  "videos": [
    {
      "device": "prevas-test-01",
      "path": "prevas-test-01/2026-07-20T12-30-00Z.mp4",
      "size": 1048576,
      "created_at": "2026-07-20T12:30:05.000Z",
      "updated_at": "2026-07-20T12:30:05.000Z"
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `device` | Device that uploaded the file |
| `path` | Storage path — use this to get a download URL |
| `size` | Size in bytes (may be null) |
| `created_at` | First created in storage (may be null) |
| `updated_at` | Last written (may be null). Changes if the same path is overwritten. |

---

## 4. Download a video

### Step A — get a signed download URL

```bash
curl -H "Authorization: Bearer $EASYSORT_API_KEY" \
  "$API_BASE/v1/vision/videos/prevas-test-01/2026-07-20T12-30-00Z.mp4"
```

Example response:

```json
{
  "path": "prevas-test-01/2026-07-20T12-30-00Z.mp4",
  "download_url": "https://….supabase.co/storage/v1/object/sign/…?token=…",
  "expires_in": 300
}
```

### Step B — GET the file from `download_url`

```bash
curl -o clip.mp4 "$DOWNLOAD_URL"
```

Optional: follow a redirect instead of parsing JSON:

```bash
curl -L -o clip.mp4 \
  -H "Authorization: Bearer $EASYSORT_API_KEY" \
  "$API_BASE/v1/vision/videos/prevas-test-01/2026-07-20T12-30-00Z.mp4?redirect=true"
```

---

## 5. Suggested smoke test

```bash
export API_BASE='https://api.easysort.org'
export DEVICE_TOKEN='<YOUR_DEVICE_TOKEN>'
export EASYSORT_API_KEY='<YOUR_API_KEY>'

printf 'test' > /tmp/vision-smoke.mp4

RESP="$(curl -sS -X POST \
  -H "Authorization: Bearer $DEVICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"filename":"smoke-test.mp4"}' \
  "$API_BASE/v1/vision/upload-url")"
UPLOAD_URL="$(printf '%s' "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["upload_url"])')"
PATH_IN_BUCKET="$(printf '%s' "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])')"

curl -sS -X PUT -H "Content-Type: video/mp4" --data-binary @/tmp/vision-smoke.mp4 "$UPLOAD_URL"

curl -sS -H "Authorization: Bearer $EASYSORT_API_KEY" "$API_BASE/v1/vision/videos"

DL="$(curl -sS -H "Authorization: Bearer $EASYSORT_API_KEY" "$API_BASE/v1/vision/videos/$PATH_IN_BUCKET")"
DOWNLOAD_URL="$(printf '%s' "$DL" | python3 -c 'import json,sys; print(json.load(sys.stdin)["download_url"])')"
curl -sS -o /tmp/vision-downloaded.mp4 "$DOWNLOAD_URL"
```

---

## 6. Errors

| Status | Meaning | What to do |
|---|---|---|
| `200` | Success | — |
| `400` | Bad filename / path | Use a simple `*.mp4` name / path from list |
| `401` | Missing / invalid token or API key | Check the Authorization header |
| `403` | API key has no Vision+ access, or device is blocked | Contact Easysort |
| `404` | Unknown path | List videos again |
| `503` | Temporary issue minting URLs | Retry with backoff |

If the signed URL itself fails (expired / already used incorrectly), request a new URL from the API.

---

## 7. Notes

- Paths look like `{device}/{filename}.mp4`
- Device token ≠ API key
- Signed URLs expire in 5 minutes (`expires_in: 300`)
- Please cache downloads you need; a handful of API requests per minute is plenty
- Questions, extra devices, or key/token rotation? Contact Easysort

# Easysort Vision+ API — Customer Guide

This guide explains how to **list** and **download** videos uploaded by IoT devices into
Easysort Vision+.

- **Base URL:** `https://api.easysort.org` (confirm with Easysort if different)
- **Auth:** every request below needs  
  `Authorization: Bearer <YOUR_API_KEY>`
- You only receive a key if Vision+ is enabled for your organisation.

> Keep your API key secret. Anyone with the key can list and download all Vision+ videos.
> If it leaks, tell Easysort and we will rotate it.

---

## 1. Health check (optional)

```bash
curl https://api.easysort.org/v1/health
```

```json
{ "status": "ok" }
```

---

## 2. List videos

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
      "updated_at": "2026-07-20T12:30:05.000Z"
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `device` | Device that uploaded the file |
| `path` | Bucket path — use this to download |
| `size` | Size in bytes (may be null) |
| `updated_at` | Last update timestamp from storage (may be null) |

---

## 3. Download a video

Pass the `path` from the list response. The body is the MP4 file.

```bash
curl -H "Authorization: Bearer $EASYSORT_API_KEY" \
  -o clip.mp4 \
  "$API_BASE/v1/vision/videos/prevas-test-01/2026-07-20T12-30-00Z.mp4"
```

Then open `clip.mp4` to verify the upload.

---

## 4. Errors

| Status | Meaning | What to do |
|---|---|---|
| `200` | Success | — |
| `401` | Missing / invalid API key | Check the `Authorization` header |
| `403` | Key has no Vision+ access | Contact Easysort |
| `404` | Unknown path | List videos again |
| `503` | Temporary storage issue | Retry with backoff |

Error body example:

```json
{ "detail": "Invalid API key." }
```

---

## 5. Notes

- Paths look like `{device}/{filename}.mp4`.
- Please cache downloads you need; avoid hammering the list endpoint.
- A handful of requests per minute is plenty.
- Questions or key rotation? Contact Easysort.

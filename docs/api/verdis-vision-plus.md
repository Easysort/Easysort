# Easysort Vision+ API — Customer Guide

This guide covers how to:

1. **Upload** MP4 videos from a device (device token)
2. **List** and **download** those videos (API key) to verify they arrived

- **Base URL:** `https://api.easysort.org` (confirm with Easysort if different)
- You receive two kinds of secret from Easysort:
  - **Device token** — used only for upload from a device
  - **API key** — used to list and download videos

> Treat both like passwords. Anyone with the device token can upload as that device;
> anyone with the API key can list and download all Vision+ videos. If either leaks,
> tell Easysort and we will rotate it.

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

Each device has its own **device token** and a fixed **device name** (e.g. `prevas-test-01`).
Uploads are stored as `{device_name}/{filename}.mp4`.

```bash
export DEVICE_TOKEN='<YOUR_DEVICE_TOKEN>'
export API_BASE='https://api.easysort.org'

curl -X POST \
  -H "Authorization: Bearer $DEVICE_TOKEN" \
  -F "file=@clip.mp4" \
  "$API_BASE/v1/vision/upload"
```

Optional: choose the stored filename (must end in `.mp4`):

```bash
curl -X POST \
  -H "Authorization: Bearer $DEVICE_TOKEN" \
  -F "file=@clip.mp4" \
  -F "filename=2026-07-20T12-30-00Z.mp4" \
  "$API_BASE/v1/vision/upload"
```

Example response:

```json
{
  "device": "prevas-test-01",
  "path": "prevas-test-01/clip.mp4",
  "size": 1048576
}
```

### Upload rules

- File must be an **MP4** (`.mp4`)
- Maximum size **200 MiB**
- Filename: simple characters only (letters, digits, `.`, `_`, `-`)
- Uploading the same path again **overwrites** the previous file
- Put the device token only on the device (or a secure secret store) — not in public repos

### Typical device integration

Call the upload endpoint after each recording finishes (or on a schedule). Example sketch:

```bash
# After recording /path/to/recording.mp4 on the device:
curl -X POST \
  -H "Authorization: Bearer $DEVICE_TOKEN" \
  -F "file=@/path/to/recording.mp4" \
  -F "filename=$(date -u +%Y-%m-%dT%H-%M-%SZ).mp4" \
  "https://api.easysort.org/v1/vision/upload"
```

Need another device (extra token / device name)? Ask Easysort.

---

## 3. List videos

Use your **API key** (not the device token):

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

## 4. Download a video

Pass the `path` from the list response. The body is the MP4 file.

```bash
curl -H "Authorization: Bearer $EASYSORT_API_KEY" \
  -o clip.mp4 \
  "$API_BASE/v1/vision/videos/prevas-test-01/2026-07-20T12-30-00Z.mp4"
```

Open `clip.mp4` locally to verify the upload.

---

## 5. Suggested smoke test

After you receive credentials:

```bash
export API_BASE='https://api.easysort.org'
export DEVICE_TOKEN='<YOUR_DEVICE_TOKEN>'
export EASYSORT_API_KEY='<YOUR_API_KEY>'

# 1) Upload a small test file
printf 'test' > /tmp/vision-smoke.mp4
curl -X POST \
  -H "Authorization: Bearer $DEVICE_TOKEN" \
  -F "file=@/tmp/vision-smoke.mp4" \
  -F "filename=smoke-test.mp4" \
  "$API_BASE/v1/vision/upload"

# 2) Confirm it appears in the list
curl -H "Authorization: Bearer $EASYSORT_API_KEY" \
  "$API_BASE/v1/vision/videos?device=prevas-test-01"

# 3) Download it back
curl -H "Authorization: Bearer $EASYSORT_API_KEY" \
  -o /tmp/vision-downloaded.mp4 \
  "$API_BASE/v1/vision/videos/prevas-test-01/smoke-test.mp4"
```

---

## 6. Errors

| Status | Meaning | What to do |
|---|---|---|
| `200` | Success | — |
| `400` | Bad filename / empty file / not MP4 | Check file and filename |
| `401` | Missing / invalid token or API key | Check the `Authorization` header |
| `403` | API key has no Vision+ access, or device is blocked | Contact Easysort |
| `404` | Unknown download path | List videos again |
| `413` | File larger than 200 MiB | Send a smaller clip |
| `503` | Temporary storage issue | Retry with backoff |

Error body example:

```json
{ "detail": "Invalid API key." }
```

---

## 7. Notes

- Paths look like `{device}/{filename}.mp4`.
- Device token ≠ API key — use the right one for upload vs list/download.
- Please cache downloads you need; avoid hammering the list endpoint.
- A handful of requests per minute is plenty.
- Questions, extra devices, or key/token rotation? Contact Easysort.

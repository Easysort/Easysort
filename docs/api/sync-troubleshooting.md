# SYNC Troubleshooting Guide

This document explains the `Registry.SYNC()` procedure and documents potential issues with file deletion.

---

## Current SYNC Procedure

The `SYNC()` method performs the following steps:

### Step 1: List Remote Files
```
dirs = [top-level directories from Supabase bucket]
files = []
while dirs not empty:
    cur = dirs.pop()
    paths = list contents of cur
    dirs += [subdirectories (no "." in name)]
    files += [files (has "." in name)]
```

### Step 2: Download Missing Files
- Compares remote files against local registry
- **Excludes `.jpg` files** from the missing files check
- Downloads missing files in parallel using `thread_map`
- Writes missing file list to `missing_files.txt`

### Step 3: Health Check
- Calls `self.is_healthy()` to verify registry integrity
- Fails sync if registry is unhealthy

### Step 4: Cleanup Old Videos
- Iterates over **all** remote files
- Parses timestamp from file path structure:
  ```
  path.split("/")[-5] → year
  path.split("/")[-4] → month  
  path.split("/")[-3] → day
  path.split("/")[-1][:2] → hour
  path.split("/")[-1][2:4] → minute
  path.split("/")[-1][4:6] → second
  ```
- Deletes files older than 2 weeks (in batches of 100)

---

## Potential Issues with File Deletion

### Issue 1: `is_healthy()` Method Not Defined

**Problem:** Line 53 calls `self.is_healthy()` but this method is not defined in `RegistryBase`.

```python
assert self.is_healthy(), "Registry is not healthy"  # ← AttributeError
```

**Impact:** SYNC will crash before reaching the cleanup phase, so no files get deleted.

**Solution:** Either implement `is_healthy()` or remove this check.

---

### Issue 2: JPG Files Have Different Path Structure

**Problem:** The timestamp parsing assumes a specific path structure like:
```
argo/device/2025/12/10/082225.mp4
     [-6]  [-5] [-4][-3] [-2]   [-1]
```

But JPG files may have a different structure like:
```
argo/device/2025/12/10/08/photo_20251210T082225Z.jpg
```

**Impact:** Parsing `file.split("/")[-1][:2]` on a JPG filename like `photo_20251210T082225Z.jpg` returns `"ph"` instead of the hour, causing `int("ph")` to raise `ValueError`.

**Solution:** Add try/except around timestamp parsing, or filter out JPG files before cleanup:
```python
if ".jpg" in file: 
    continue
```

---

### Issue 3: No Error Handling in Timestamp Parsing

**Problem:** The cleanup loop has no error handling:

```python
year, month, day, hour, minute, second = file.split("/")[-5], ...
timestamp = datetime.datetime(int(year), int(month), int(day), ...)
```

**Impact:** Any file with an unexpected path structure will crash the entire SYNC with `ValueError` or `IndexError`, preventing deletion of any files.

**Solution:** Wrap in try/except:
```python
try:
    year, month, day = file.split("/")[-5], file.split("/")[-4], file.split("/")[-3]
    ...
except (ValueError, IndexError):
    continue  # Skip files that don't match expected format
```

---

### Issue 4: Directory Detection Uses "." Heuristic

**Problem:** The code distinguishes files from directories by checking for `.` in the name:

```python
dirs.extend([cur / Path(x) for x in paths if not "." in x])  # directories
files.extend([cur / Path(x) for x in paths if "." in x])      # files
```

**Impact:** 
- A directory named `v1.0` would be treated as a file
- A file without an extension would be treated as a directory

**Solution:** Use Supabase metadata to check if entry is a file or directory, or use more robust detection.

---

### Issue 5: Files Deleted From Remote But Not Local

**Problem:** SYNC deletes files from the **Supabase remote storage** only. The local registry copy remains.

```python
supabase_client.storage.from_(SUPABASE_DATA_REGISTRY_BUCKET).remove(files_to_delete[...])
```

**Impact:** Local disk fills up with old files that were deleted from remote. Future syncs won't re-download these files (they already exist locally), but they also won't delete them.

**Solution:** Also delete from local registry:
```python
for file in files_to_delete:
    local_path = Path(self.registry_path) / SUPABASE_DATA_REGISTRY_BUCKET / file
    if local_path.exists():
        local_path.unlink()
```

---

## Recommended Fix

Here's a corrected cleanup section:

```python
print("Cleanup videos older than 2 weeks")
files_to_delete = []
for file in tqdm(files, desc="Cleanup videos"):
    file = str(file)
    
    # Skip image files unless explicitly allowed
    if ".jpg" in file:
        continue
    
    try:
        year = file.split("/")[-5]
        month = file.split("/")[-4]
        day = file.split("/")[-3]
        hour = file.split("/")[-1][:2]
        minute = file.split("/")[-1][2:4]
        second = file.split("/")[-1][4:6]
        timestamp = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        if timestamp < datetime.datetime.now() - datetime.timedelta(weeks=2):
            files_to_delete.append(file)
    except (ValueError, IndexError):
        continue  # Skip files that don't match expected format

print(f"Deleting {len(files_to_delete)} files" if files_to_delete else "No files to delete")
if not files_to_delete:
    return

for i in tqdm(range(0, len(files_to_delete), 100), desc="Deleting files"):
    batch = files_to_delete[i:i+100]
    supabase_client.storage.from_(SUPABASE_DATA_REGISTRY_BUCKET).remove(batch)
    time.sleep(1)
```

---

## Debugging Steps

1. **Check if SYNC completes:** Look for "Sync complete" and "Cleanup videos" messages
2. **Check `missing_files.txt`:** See what files were considered missing
3. **Test timestamp parsing:** Run the parsing logic on a sample of your file paths
4. **Check Supabase logs:** Verify delete requests are being sent and succeeding
5. **Compare local vs remote:** After SYNC, compare file counts locally vs in Supabase dashboard

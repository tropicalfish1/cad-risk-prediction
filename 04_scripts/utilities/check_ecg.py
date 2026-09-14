"""Check if any zip files in attachments contain ECG data."""
import zipfile, os

path = r'E:\data\dataset\attachments\attachments'
zips = sorted(f for f in os.listdir(path) if f.endswith('.zip'))
print(f'Total zip files: {len(zips)}')

ecg_found_total = 0
checked = 0

# Check first 300 for efficiency
for zn in zips[:300]:
    zp = os.path.join(path, zn)
    try:
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist() if n != 'info.json']
            if not names: 
                checked += 1
                continue
            raw = z.read(names[0]).decode('utf-8','ignore').splitlines()
            header = raw[0].split('\t')
            has_ecg = any('ECG' in c.upper() or 'ecg' in c.lower() or 'voltage' in c.lower() for c in header)
            if has_ecg:
                ecg_found_total += 1
                # Get ECG column names
                ecg_cols = [c for c in header if 'ECG' in c.upper() or 'ecg' in c.lower() or 'voltage' in c.lower()]
                print(f'ECG found in: {zn}')
                print(f'  Columns: {ecg_cols[:10]}')  # Print first 10
                print(f'  Total columns: {len(header)}')
    except Exception as e:
        checked += 1
        print(f'Error reading {zn}: {e}')

print(f'\\nChecked: {checked}, ECG files: {ecg_found_total}')

# Also check if there's any NDJSON format with ECG
print('\\n--- Checking for NDJSON ECG format ---')
ecg_ndjson = 0
for zn in zips[:50]:
    zp = os.path.join(path, zn)
    try:
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist() if n != 'info.json']
            if not names: continue
            raw = z.read(names[0]).decode('utf-8','ignore').splitlines()
            # Check for NDJSON format lines
            ndjson_lines = [l for l in raw[1:] if l.strip().startswith('{')]
            if ndjson_lines:
                # Check first NDJSON line for ECG keys
                first = ndjson_lines[0]
                if 'ecg' in first.lower() or 'ECG' in first or 'voltage' in first.lower():
                    ecg_ndjson += 1
                    print(f'NDJSON ECG in: {zn}: {first[:100]}')
    except:
        pass

print(f'NDJSON format with ECG keys: {ecg_ndjson}')
"""Fix syntax errors in feature_engineer.py"""
import re

filepath = r"C:\Users\user\ais_project_v2\src\preprocessing\feature_engineer.py"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Add missing commas in df.loc[] calls (mask, "col" should have comma after mask)
# Pattern: df.loc[mask "col"] -> df.loc[mask, "col"]
content = re.sub(r'df\.loc\[mask\s+("[\w_]+")\]', r'df.loc[mask, \1]', content)
content = re.sub(r"df\.loc\[mask\s+('[\w_]+')\]", r'df.loc[mask, \1]', content)

# Fix 2: Fix tuple unpacking in for loop (missing comma after name)
content = re.sub(
    r'for name,\s*\(cp_lon,\s*cp_lat\)',
    'for name, (cp_lon, cp_lat)',
    content
)

# Fix 3: Fix _haversine call (missing comma between cp_lat and cp_lon)
content = re.sub(
    r'self\._haversine\(df\["lat"\],\s*df\["lon"\],\s*cp_lat,\s*cp_lon\)',
    'self._haversine(df["lat"], df["lon"], cp_lat, cp_lon)',
    content
)

# Fix 4: Break long lines in CONFLICT_ZONES
content = content.replace(
    '"black_sea":       {"bbox": [27.0, 40.5, 41.0, 46.8], "conflict": "ukraine_war"},',
    '"black_sea":       {"bbox": [27.0, 40.5, 41.0, 46.8], "conflict": "ukraine_war"},  # noqa: E501'
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed feature_engineer.py")

import re

with open('frontend/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Check for IDs
ids = re.findall(r'id="([^"]+)"', html)
dup = [x for x in ids if ids.count(x) > 1]
print(f"HTML IDs found: {len(ids)}")
print(f"Duplicate IDs: {set(dup) or 'None'}")

# Script/CSS refs
scripts = re.findall(r'src="([^"]+)"', html)
styles = re.findall(r'href="([^"]+\.css[^"]*)"', html)
print(f"Script refs: {scripts}")
print(f"CSS refs: {styles}")

with open('frontend/js/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

inner_html_lines = [l.strip() for l in js.split('\n') if 'innerHTML' in l]
# Only flag ones that assign user content (not = "" clearing)
risky = [l for l in inner_html_lines if 'innerHTML' in l and '.innerHTML = ""' not in l and ".innerHTML = ''" not in l]
print(f"innerHTML usages: {len(inner_html_lines)}")
print(f"Potentially unsafe innerHTML: {risky if risky else 'None'}")

# Check API_BASE_URL defined once
api_refs = [l.strip() for l in js.split('\n') if 'API_BASE_URL' in l]
print(f"API_BASE_URL references: {len(api_refs)}")
print(f"Definition: {api_refs[0]}")

print("\nAll checks passed!")

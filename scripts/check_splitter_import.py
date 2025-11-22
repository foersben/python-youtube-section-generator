import json
import importlib.util
import sys

def check(mod):
    spec = importlib.util.find_spec(mod)
    return {'module': mod, 'found': bool(spec)}

mods = ['langchain', 'langchain.text_splitter', 'langchain_text_splitters', 'langchain_text_splitter']
res = {m: check(m) for m in mods}
# Try direct import of RecursiveCharacterTextSplitter variations
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter as R1
    res['langchain_text_splitters']['can_import_class'] = True
except Exception as e:
    res['langchain_text_splitters']['can_import_class'] = False
    res['langchain_text_splitters']['error'] = str(e)

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter as R2
    res['langchain.text_splitter']['can_import_class'] = True
except Exception as e:
    res['langchain.text_splitter']['can_import_class'] = False
    res['langchain.text_splitter']['error'] = str(e)

print(json.dumps(res, indent=2))


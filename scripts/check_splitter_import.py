"""Check the importability of various langchain text splitter modules.

Attributes:
    - module: The module name being checked.
    - found: Boolean indicating if the module was found.
    - can_import_class: Boolean indicating if RecursiveCharacterTextSplitter can be imported (if applicable).
    - error: Error message if import failed (if applicable).
"""

import importlib.util
import json


def check(mod: str) -> dict[str, bool | str]:
    """Check if a module can be found and imported.

    Args:
        mod (str): The module name to check.

    Returns:
        dict: A dictionary with module name and whether it was found.
    """

    spec = importlib.util.find_spec(mod)
    return {"module": mod, "found": bool(spec)}


mods = [
    "langchain",
    "langchain.text_splitter",
    "langchain_text_splitters",
    "langchain_text_splitter",
]
res = {m: check(m) for m in mods}
# Try direct import of RecursiveCharacterTextSplitter variations
try:

    res["langchain_text_splitters"]["can_import_class"] = True
except Exception as e:
    res["langchain_text_splitters"]["can_import_class"] = False
    res["langchain_text_splitters"]["error"] = str(e)

print(json.dumps(res, indent=2))

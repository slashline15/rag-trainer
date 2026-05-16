import sys
sys.path.insert(0, "src")
from rag_sqlite.tg.helpers import markdown_to_html

# Caso 1: bloco de codigo python
r = markdown_to_html("Exemplo:\n```python\nprint('hello')\n```\nFim.")
assert "<pre><code" in r, f"FAIL code block: {r}"
assert "CODEBLOCK" not in r, f"FAIL placeholder leaked: {r}"
print("OK code block:", r[:100])

# Caso 2: inline code
r = markdown_to_html("Use `git status` para ver.")
assert "<code>" in r, f"FAIL inline code: {r}"
assert "INLINE" not in r, f"FAIL inline placeholder: {r}"
print("OK inline code:", r[:80])

# Caso 3: codigo sem linguagem
r = markdown_to_html("```\nSELECT * FROM chunks;\n```")
assert "CODEBLOCK" not in r
assert "<pre>" in r
print("OK no-lang block:", r[:80])

# Caso 4: bold nao quebra codigo
r = markdown_to_html("**titulo**\n```sql\nSELECT 1;\n```")
assert "<b>titulo</b>" in r, f"FAIL bold: {r}"
assert "CODEBLOCK" not in r, f"FAIL placeholder in bold+code: {r}"
print("OK bold+code:", r[:100])

# Caso 5: multiplos blocos
r = markdown_to_html("```python\nx=1\n```\ntexto\n```json\n{}\n```")
assert r.count("</pre>") == 2, f"FAIL multi-block count: {r}"
assert "CODEBLOCK" not in r
print("OK multi-block:", r[:150])

print("\nTODOS OK")

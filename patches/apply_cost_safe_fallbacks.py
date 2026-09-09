from pathlib import Path

path = Path("src/clients/model_router.py")
text = path.read_text(encoding="utf-8")

old_map = '''CAPABILITY_MAP: Dict[str, List[Tuple[str, str]]] = {
    "fast": [
        ("x-ai/grok-4.1-fast", "openrouter"),
        ("google/gemini-3.1-pro-preview", "openrouter"),
    ],
    "cheap": [
        ("deepseek/deepseek-v3.2", "openrouter"),
        ("google/gemini-3.1-pro-preview", "openrouter"),
    ],
'''
new_map = '''CAPABILITY_MAP: Dict[str, List[Tuple[str, str]]] = {
    "fast": [
        ("x-ai/grok-4.1-fast", "openrouter"),
        ("deepseek/deepseek-v3.2", "openrouter"),
    ],
    "cheap": [
        ("deepseek/deepseek-v3.2", "openrouter"),
        ("x-ai/grok-4.1-fast", "openrouter"),
    ],
'''

old_resolve = '''        if model is not None:
            provider = self._infer_provider(model)
            targets.append((model, provider))
            fleet = self._fleet_for_provider(provider)
        elif capability is not None:
            cap_targets = self._active_capability_map().get(capability, [])
            targets.extend(cap_targets)
        else:
            targets = list(fleet)

        # Append remaining fleet members not yet in the list
'''
new_resolve = '''        if model is not None:
            provider = self._infer_provider(model)
            targets.append((model, provider))
            if provider == "openrouter" and model in {
                "x-ai/grok-4.1-fast",
                "deepseek/deepseek-v3.2",
            }:
                fleet = list(CAPABILITY_MAP["cheap"])
            else:
                fleet = self._fleet_for_provider(provider)
        elif capability is not None:
            cap_targets = self._active_capability_map().get(capability, [])
            targets.extend(cap_targets)
            if self.default_provider == "openrouter" and capability in {"fast", "cheap"}:
                fleet = list(cap_targets)
        else:
            targets = list(fleet)

        # Append remaining fleet members not yet in the list
'''

for label, old, new in [
    ("capability map", old_map, new_map),
    ("target resolution", old_resolve, new_resolve),
]:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Expected exactly one {label} block, found {count}")
    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
print("Cost-safe fallback routing updated exactly once per target block.")

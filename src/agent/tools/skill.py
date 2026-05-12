from agent import info


def load_skill(name: str) -> str:
    body = info.load_skill(name)
    if body is None:
        return f"Error: skill '{name}' not found. Call list_skills() to see available skills."
    return body


def list_skills() -> str:
    meta = info.scan_skills_meta()
    if not meta:
        return "No skills available."
    return info.format_skills_listing(meta)

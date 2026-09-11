import yaml
def load_config(config_path: str) -> dict:
    """โหลด config จากไฟล์ YAML แล้ว apply ENV VAR overrides ทับ"""
    with open(config_path, "r", encoding="utf-8-sig") as config_file:
        config = yaml.safe_load(config_file)
    return config or {}
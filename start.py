from __future__ import annotations

import sys


def main() -> None:
    try:
        import uvicorn
    except ImportError:
        print(
            "Uvicorn is missing. Install dependencies with: pip install -r requirements.txt", file=sys.stderr
        )
        raise SystemExit(1)
    from app.config import get_settings

    settings = get_settings()
    print("Starting Orbit Intelligent Assistant")
    print(f"Open http://{settings.app_host}:{settings.app_port} in your browser")
    missing_runtime = []
    import importlib.util

    for module, label in (
        ("litellm", "LiteLLM"),
        ("chromadb", "ChromaDB"),
        ("google.genai", "Google GenAI SDK"),
    ):
        try:
            installed = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        if not installed:
            missing_runtime.append(label)
    if missing_runtime:
        print("Missing runtime dependencies: " + ", ".join(missing_runtime))
        print("Install them with: python -m pip install -r requirements.txt")
    if issues := settings.readiness_issues():
        print("Provider setup required:")
        for issue in issues:
            print(f"  - {issue}")
        print(
            "The setup UI will still load; chat becomes ready after .env is configured and the server restarts."
        )
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

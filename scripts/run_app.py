"""
App bootstrap and runner script.
Checks configuration, installs requirements (optional), and runs services.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


def check_env():
    """Ensure .env file exists or copy from example."""
    env_file = ROOT_DIR / ".env"
    example_file = ROOT_DIR / ".env.example"

    if not env_file.exists():
        print(f"⚠️  .env file not found. Copying from {example_file.name}...")
        if example_file.exists():
            import shutil
            shutil.copy(str(example_file), str(env_file))
            print("✅ Created .env file. Please edit it with your GEMINI_API_KEY.")
        else:
            print("❌ .env.example file not found!")
            sys.exit(1)


def run_backend(port=8000):
    """Start the FastAPI backend server."""
    print(f"🚀 Starting FastAPI backend on port {port}...")
    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", str(port), "--reload"],
            cwd=str(ROOT_DIR),
            check=True
        )
    except KeyboardInterrupt:
        print("\nStopping backend server.")


def run_frontend(port=8501, backend_url="http://localhost:8000"):
    """Start the Streamlit frontend app."""
    print(f"🚀 Starting Streamlit frontend on port {port}...")
    os.environ["BACKEND_URL"] = backend_url
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", str(port)],
            cwd=str(ROOT_DIR),
            check=True
        )
    except KeyboardInterrupt:
        print("\nStopping frontend server.")


def main():
    parser = argparse.ArgumentParser(description="Run RAG AI Teaching Assistant services")
    parser.add_argument("service", choices=["backend", "frontend", "all"], help="Service to run")
    parser.add_argument("--backend-port", type=int, default=8000, help="Backend port")
    parser.add_argument("--frontend-port", type=int, default=8501, help="Frontend port")
    parser.add_argument("--backend-url", type=str, default="http://localhost:8000", help="URL of the running backend")

    args = parser.parse_args()

    check_env()

    if args.service == "backend":
        run_backend(args.backend_port)
    elif args.service == "frontend":
        run_frontend(args.frontend_port, args.backend_url)
    elif args.service == "all":
        print("⚠️ To run both services, please run them in separate terminals or use docker-compose.")
        print(f"Run backend: python scripts/run_app.py backend --backend-port {args.backend_port}")
        print(f"Run frontend: python scripts/run_app.py frontend --frontend-port {args.frontend_port} --backend-url http://localhost:{args.backend_port}")


if __name__ == "__main__":
    main()

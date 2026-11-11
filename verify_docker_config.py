"""
verify_docker_config.py - Static configuration validator for PrivSynth Stage 3 Docker & Nginx assets.
"""
import os
import re

def verify_stage3():
    print("=== Validating Stage 3 Docker & Nginx Configurations ===")
    errors = []

    # 1. Dockerfile Validation
    print("1. Validating Dockerfile...")
    if not os.path.exists("Dockerfile"):
        errors.append("Dockerfile does not exist.")
    else:
        with open("Dockerfile", "r", encoding="utf-8") as f:
            df_content = f.read()
        assert "python:3.13-slim" in df_content, "Base image must be python:3.13-slim"
        assert "USER appuser" in df_content, "Must run as non-root user"
        assert "HOST=0.0.0.0" in df_content, "HOST must be set to 0.0.0.0"
        assert "--workers 1" in df_content, "Uvicorn must use exactly 1 worker"
        assert "--reload" not in df_content, "Uvicorn must not use --reload in production"
        assert "HEALTHCHECK" in df_content, "Dockerfile must define a HEALTHCHECK"
        assert "requirements.txt" in df_content, "Must install from requirements.txt"
        print("   -> Dockerfile passed all static checks.")

    # 2. .dockerignore Validation
    print("2. Validating .dockerignore...")
    if not os.path.exists(".dockerignore"):
        errors.append(".dockerignore does not exist.")
    else:
        with open(".dockerignore", "r", encoding="utf-8") as f:
            di_content = f.read()
        for excluded in [".git/", ".venv/", "__pycache__/", "notebooks/", ".env", "test_*.py"]:
            assert excluded in di_content, f".dockerignore must exclude {excluded}"
        lines = [line.strip() for line in di_content.splitlines()]
        assert "backend/" not in lines and "backend" not in lines, "backend/ must NOT be ignored"
        assert "privsynth/" not in lines and "privsynth" not in lines, "privsynth/ must NOT be ignored"
        print("   -> .dockerignore passed all static checks.")

    # 3. Nginx Configuration Validation
    print("3. Validating nginx/default.conf...")
    if not os.path.exists("nginx/default.conf"):
        errors.append("nginx/default.conf does not exist.")
    else:
        with open("nginx/default.conf", "r", encoding="utf-8") as f:
            nginx_conf = f.read()
        assert "proxy_pass http://backend_service;" in nginx_conf, "Must proxy to backend_service"
        assert "client_max_body_size 60M;" in nginx_conf, "client_max_body_size must accommodate 50MB payload"
        assert "proxy_read_timeout 300s;" in nginx_conf, "proxy_read_timeout must accommodate CTGAN generation"
        assert "proxy_send_timeout 300s;" in nginx_conf, "proxy_send_timeout must accommodate CTGAN generation"
        assert "gzip on;" in nginx_conf, "gzip compression must be enabled"
        assert "/usr/share/nginx/html" in nginx_conf, "Static root must point to /usr/share/nginx/html"
        assert "=404" in nginx_conf, "Must have 404 fallback for missing static files"
        print("   -> nginx/default.conf passed all static checks.")

    # 4. docker-compose.yml Validation
    print("4. Validating docker-compose.yml...")
    if not os.path.exists("docker-compose.yml"):
        errors.append("docker-compose.yml does not exist.")
    else:
        with open("docker-compose.yml", "r", encoding="utf-8") as f:
            compose_content = f.read()
        # Verify YAML structure with PyYAML
        try:
            import yaml
            parsed = yaml.safe_load(compose_content)
            assert "services" in parsed, "services must be defined"
            assert "backend" in parsed["services"], "backend service must be defined"
            assert "nginx" in parsed["services"], "nginx service must be defined"
            assert "volumes" in parsed, "volumes must be defined"
            assert "privsynth_storage" in parsed["volumes"], "privsynth_storage volume must be defined"

            # Check port exposure: only nginx has published ports
            backend_svc = parsed["services"]["backend"]
            nginx_svc = parsed["services"]["nginx"]
            assert "ports" not in backend_svc, "Backend port must NOT be published to host"
            assert "expose" in backend_svc and "8000" in [str(p) for p in backend_svc["expose"]], "Backend must expose internal 8000"
            assert "ports" in nginx_svc, "Nginx must expose public port"

            # Check volume mount
            backend_vols = [str(v) for v in backend_svc.get("volumes", [])]
            assert any("privsynth_storage:/app/storage" in v for v in backend_vols), "Backend must mount privsynth_storage to /app/storage"

            # Check healthcheck and dependency
            assert "healthcheck" in backend_svc, "Backend must define healthcheck"
            assert "depends_on" in nginx_svc, "Nginx must depend on backend"
            print("   -> docker-compose.yml parsed successfully and passed all static assertions.")
        except ImportError:
            print("   (PyYAML not installed, performing regex verification)")
            assert "privsynth_storage:/app/storage" in compose_content
            assert "expose:" in compose_content

    # 5. Frontend API Base URL Validation
    print("5. Validating frontend/js/app.js API_BASE_URL resolution...")
    with open("frontend/js/app.js", "r", encoding="utf-8") as f:
        app_js = f.read()
    assert 'window.location.port === "5500"' in app_js, "app.js must preserve port 5500 dev support"
    print("   -> frontend/js/app.js passed URL resolution check.")

    print("\n=======================================================")
    if errors:
        print(f"FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f" - {e}")
        return False
    else:
        print("ALL DOCKER & NGINX STATIC VALIDATION CHECKS PASSED!")
        print("=======================================================")
        return True

if __name__ == "__main__":
    success = verify_stage3()
    exit(0 if success else 1)

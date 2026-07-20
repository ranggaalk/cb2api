#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE="${DEPLOY_ENV_FILE:-.env}"
DEPLOY_PORT="${DEPLOY_PORT:-8001}"
CLIENT_AUTH_MODE="${DEPLOY_CLIENT_AUTH_MODE:-passthrough}"
UPSTREAM_HEADER="${DEPLOY_UPSTREAM_HEADER:-bearer}"
RELAY_PASSWORD="${DEPLOY_RELAY_PASSWORD:-}"
ADMIN_PASSWORD="${DEPLOY_ADMIN_PASSWORD:-}"
HEALTH_TIMEOUT="${DEPLOY_HEALTH_TIMEOUT:-90}"
SKIP_DOCKER_INSTALL="${SKIP_DOCKER_INSTALL:-0}"
NO_BUILD="${DEPLOY_NO_BUILD:-0}"
COMPOSE=()
ENV_CREATED=0
BOOTSTRAP_ONLY="${DEPLOY_BOOTSTRAP_ONLY:-0}"

if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    RESET='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    RESET=''
fi

info() { printf '%b[INFO]%b %s\n' "$BLUE" "$RESET" "$*"; }
success() { printf '%b[OK]%b %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '%b[WARN]%b %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die() { printf '%b[ERROR]%b %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

on_error() {
    local exit_code=$?
    printf '%b[ERROR]%b Deployment stopped at line %s (exit %s).\n' \
        "$RED" "$RESET" "${BASH_LINENO[0]:-unknown}" "$exit_code" >&2
    exit "$exit_code"
}
trap on_error ERR

usage() {
    cat <<'EOF'
Usage: ./deploy.sh [options]

One-command Ubuntu/Debian deployment for CodeBuddy2API.
Existing .env, API key files, and credentials are never overwritten.

Options:
  --env-file PATH             Environment file (default: .env)
  --port PORT                 Host port (default: 8001)
  --client-auth-mode MODE     relay, passthrough, or hybrid
  --upstream-header MODE      bearer, x-api-key, or both
  --relay-password VALUE      Set relay password on first env creation
  --admin-password VALUE      Set admin password on first env creation
  --timeout SECONDS           Health-check timeout (default: 90)
  --no-build                  Start without rebuilding the image
  --skip-docker-install       Fail instead of installing missing Docker
  --bootstrap-only            Create/validate files, then stop before Docker
  -h, --help                  Show this help

Environment equivalents:
  DEPLOY_ENV_FILE, DEPLOY_PORT, DEPLOY_CLIENT_AUTH_MODE,
  DEPLOY_UPSTREAM_HEADER, DEPLOY_RELAY_PASSWORD, DEPLOY_ADMIN_PASSWORD,
  DEPLOY_HEALTH_TIMEOUT, DEPLOY_NO_BUILD=1, SKIP_DOCKER_INSTALL=1,
  DEPLOY_BOOTSTRAP_ONLY=1
EOF
}

require_value() {
    [[ $# -ge 2 && -n "${2:-}" ]] || die "Option $1 requires a value."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-file)
            require_value "$@"; ENV_FILE="$2"; shift 2 ;;
        --port)
            require_value "$@"; DEPLOY_PORT="$2"; shift 2 ;;
        --client-auth-mode)
            require_value "$@"; CLIENT_AUTH_MODE="$2"; shift 2 ;;
        --upstream-header)
            require_value "$@"; UPSTREAM_HEADER="$2"; shift 2 ;;
        --relay-password)
            require_value "$@"; RELAY_PASSWORD="$2"; shift 2 ;;
        --admin-password)
            require_value "$@"; ADMIN_PASSWORD="$2"; shift 2 ;;
        --timeout)
            require_value "$@"; HEALTH_TIMEOUT="$2"; shift 2 ;;
        --no-build)
            NO_BUILD=1; shift ;;
        --skip-docker-install)
            SKIP_DOCKER_INSTALL=1; shift ;;
        --bootstrap-only)
            BOOTSTRAP_ONLY=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            die "Unknown option: $1. Run --help for usage." ;;
    esac
done

[[ "$DEPLOY_PORT" =~ ^[0-9]+$ ]] && (( DEPLOY_PORT >= 1 && DEPLOY_PORT <= 65535 )) \
    || die "Port must be an integer between 1 and 65535."
[[ "$HEALTH_TIMEOUT" =~ ^[0-9]+$ ]] && (( HEALTH_TIMEOUT >= 1 )) \
    || die "Timeout must be a positive integer."
case "$CLIENT_AUTH_MODE" in relay|passthrough|hybrid) ;; *) die "Invalid client auth mode." ;; esac
case "$UPSTREAM_HEADER" in bearer|x-api-key|both) ;; *) die "Invalid upstream header mode." ;; esac

if [[ "$ENV_FILE" != /* ]]; then
    ENV_FILE="$SCRIPT_DIR/$ENV_FILE"
fi

run_privileged() {
    if [[ $EUID -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        die "Root or sudo is required to install Docker."
    fi
}

install_docker() {
    [[ "$SKIP_DOCKER_INSTALL" != "1" ]] \
        || die "Docker/Compose is missing and automatic installation was skipped."
    [[ -r /etc/os-release ]] || die "Cannot detect the operating system."

    # shellcheck disable=SC1091
    source /etc/os-release
    local distro="${ID:-}"
    case "$distro" in
        ubuntu|debian) ;;
        *) die "Automatic Docker installation supports Ubuntu/Debian only (detected: $distro)." ;;
    esac

    info "Installing Docker Engine and Compose from Docker's official repository..."
    run_privileged apt-get update
    run_privileged apt-get install -y ca-certificates curl gnupg
    run_privileged install -m 0755 -d /etc/apt/keyrings

    local tmp_key
    tmp_key="$(mktemp)"
    curl -fsSL "https://download.docker.com/linux/${distro}/gpg" -o "$tmp_key"
    run_privileged install -m 0644 "$tmp_key" /etc/apt/keyrings/docker.asc
    rm -f "$tmp_key"

    local architecture codename repository
    architecture="$(dpkg --print-architecture)"
    codename="${VERSION_CODENAME:-}"
    [[ -n "$codename" ]] || die "Unable to determine distribution codename."
    repository="deb [arch=${architecture} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/${distro} ${codename} stable"
    printf '%s\n' "$repository" | run_privileged tee /etc/apt/sources.list.d/docker.list >/dev/null

    run_privileged apt-get update
    run_privileged apt-get install -y \
        docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    if command -v systemctl >/dev/null 2>&1; then
        run_privileged systemctl enable --now docker
    fi
    success "Docker installation completed."
}

detect_compose() {
    if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
        COMPOSE=(docker compose)
        return
    fi
    if command -v docker-compose >/dev/null 2>&1; then
        COMPOSE=(docker-compose)
        return
    fi

    install_docker
    if docker compose version >/dev/null 2>&1; then
        COMPOSE=(docker compose)
    elif [[ $EUID -ne 0 ]] && command -v sudo >/dev/null 2>&1 \
        && sudo docker compose version >/dev/null 2>&1; then
        COMPOSE=(sudo docker compose)
    else
        die "Docker was installed, but the Compose plugin is unavailable."
    fi
}

configure_docker_access() {
    if [[ "${COMPOSE[0]}" != "docker" ]]; then
        return
    fi
    if docker info >/dev/null 2>&1; then
        return
    elif [[ $EUID -eq 0 ]]; then
        return
    elif command -v sudo >/dev/null 2>&1 && sudo docker info >/dev/null 2>&1; then
        DOCKER_PREFIX=(sudo)
        COMPOSE=(sudo docker compose)
        warn "Using sudo for Docker commands. Add your user to the docker group for passwordless operation."
    else
        die "Docker daemon is unavailable or this user cannot access it."
    fi
}

generate_secret() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 32
    elif [[ -r /dev/urandom ]] && command -v od >/dev/null 2>&1; then
        od -An -N32 -tx1 /dev/urandom | tr -d ' \n'
    else
        die "Cannot generate a secure password: install openssl."
    fi
}

set_env_value() {
    local file="$1" key="$2" value="$3" tmp
    [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || die "Invalid newline in $key."
    tmp="$(mktemp "${file}.tmp.XXXXXX")"
    if grep -qE "^[[:space:]]*${key}=" "$file"; then
        while IFS= read -r line || [[ -n "$line" ]]; do
            if [[ "$line" =~ ^[[:space:]]*${key}= ]]; then
                printf '%s=%s\n' "$key" "$value"
            else
                printf '%s\n' "$line"
            fi
        done < "$file" > "$tmp"
    else
        cp "$file" "$tmp"
        printf '\n%s=%s\n' "$key" "$value" >> "$tmp"
    fi
    chmod 600 "$tmp"
    mv -f "$tmp" "$file"
}

get_env_value() {
    local key="$1" line
    line="$(grep -E "^[[:space:]]*${key}=" "$ENV_FILE" | tail -n 1 || true)"
    [[ -n "$line" ]] || return 1
    printf '%s' "${line#*=}"
}

bootstrap_files() {
    mkdir -p config .codebuddy_creds

    if [[ ! -f config/codebuddy_api_keys.txt ]]; then
        if [[ -f config/codebuddy_api_keys.example.txt ]]; then
            cp config/codebuddy_api_keys.example.txt config/codebuddy_api_keys.txt
        else
            : > config/codebuddy_api_keys.txt
        fi
        chmod 600 config/codebuddy_api_keys.txt
        info "Created config/codebuddy_api_keys.txt."
    fi

    if [[ ! -f "$ENV_FILE" ]]; then
        [[ -f .env.example ]] || die ".env.example is missing."
        mkdir -p "$(dirname "$ENV_FILE")"
        local staged_env
        staged_env="$(mktemp "${ENV_FILE}.bootstrap.XXXXXX")"
        cp .env.example "$staged_env"
        chmod 600 "$staged_env"
        RELAY_PASSWORD="${RELAY_PASSWORD:-$(generate_secret)}"
        ADMIN_PASSWORD="${ADMIN_PASSWORD:-$(generate_secret)}"
        while [[ "$ADMIN_PASSWORD" == "$RELAY_PASSWORD" ]]; do
            ADMIN_PASSWORD="$(generate_secret)"
        done
        set_env_value "$staged_env" CODEBUDDY_PASSWORD "$RELAY_PASSWORD"
        set_env_value "$staged_env" CODEBUDDY_ADMIN_PASSWORD "$ADMIN_PASSWORD"
        set_env_value "$staged_env" CODEBUDDY_CLIENT_AUTH_MODE "$CLIENT_AUTH_MODE"
        set_env_value "$staged_env" CODEBUDDY_UPSTREAM_API_KEY_HEADER "$UPSTREAM_HEADER"
        set_env_value "$staged_env" CODEBUDDY_HOST "0.0.0.0"
        set_env_value "$staged_env" CODEBUDDY_PORT "$DEPLOY_PORT"
        mv -f "$staged_env" "$ENV_FILE"
        ENV_CREATED=1
        success "Created secure environment file: $ENV_FILE"
    else
        chmod 600 "$ENV_FILE" || warn "Could not set permission 600 on $ENV_FILE."
        info "Using existing environment file without overwriting it: $ENV_FILE"
    fi
}

validate_environment() {
    local relay admin mode header port
    relay="$(get_env_value CODEBUDDY_PASSWORD || true)"
    admin="$(get_env_value CODEBUDDY_ADMIN_PASSWORD || true)"
    mode="$(get_env_value CODEBUDDY_CLIENT_AUTH_MODE || printf 'relay')"
    header="$(get_env_value CODEBUDDY_UPSTREAM_API_KEY_HEADER || printf 'bearer')"
    port="$(get_env_value CODEBUDDY_PORT || printf '%s' "$DEPLOY_PORT")"

    case "$mode" in relay|passthrough|hybrid) ;; *) die "Invalid CODEBUDDY_CLIENT_AUTH_MODE in $ENV_FILE." ;; esac
    case "$header" in bearer|x-api-key|both) ;; *) die "Invalid CODEBUDDY_UPSTREAM_API_KEY_HEADER in $ENV_FILE." ;; esac
    [[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) \
        || die "Invalid CODEBUDDY_PORT in $ENV_FILE."

    if [[ "$mode" == "relay" || "$mode" == "hybrid" ]]; then
        [[ -n "$relay" ]] || die "CODEBUDDY_PASSWORD is required for $mode mode."
    fi
    [[ -n "$admin" || -n "$relay" ]] \
        || die "Set CODEBUDDY_ADMIN_PASSWORD or CODEBUDDY_PASSWORD in $ENV_FILE."

    case "$relay" in relay_master_secret|change_me|password) warn "CODEBUDDY_PASSWORD still uses an insecure placeholder." ;; esac
    case "$admin" in admin_secret|change_me|password) warn "CODEBUDDY_ADMIN_PASSWORD still uses an insecure placeholder." ;; esac

    DEPLOY_PORT="$port"
    CLIENT_AUTH_MODE="$mode"
}

compose() {
    local compose_env_file="$ENV_FILE"
    if [[ "$compose_env_file" == "$SCRIPT_DIR/"* ]]; then
        compose_env_file="${compose_env_file#"$SCRIPT_DIR/"}"
    fi
    CODEBUDDY_ENV_FILE="$compose_env_file" CODEBUDDY_PORT="$DEPLOY_PORT" \
        "${COMPOSE[@]}" --env-file "$ENV_FILE" "$@"
}

health_request() {
    local url="$1"
    if command -v curl >/dev/null 2>&1; then
        curl -fsS --max-time 5 "$url" >/dev/null
    elif command -v python3 >/dev/null 2>&1; then
        python3 - "$url" <<'PY' >/dev/null
import sys
import urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
    else
        return 1
    fi
}

show_diagnostics() {
    warn "Container did not become healthy."
    compose ps || true
    compose logs --tail=200 codebuddy2api || true
}

main() {
    info "Preparing CodeBuddy2API deployment in $SCRIPT_DIR"
    bootstrap_files
    validate_environment
    if [[ "$BOOTSTRAP_ONLY" == "1" ]]; then
        success "Bootstrap completed; Docker startup was skipped."
        return
    fi
    detect_compose
    configure_docker_access

    info "Validating Docker Compose configuration..."
    compose config >/dev/null

    local up_args=(up -d --remove-orphans)
    if [[ "$NO_BUILD" != "1" ]]; then
        up_args+=(--build)
    fi
    info "Building and starting CodeBuddy2API..."
    compose "${up_args[@]}"

    local health_url="http://127.0.0.1:${DEPLOY_PORT}/health"
    local deadline=$((SECONDS + HEALTH_TIMEOUT))
    info "Waiting up to ${HEALTH_TIMEOUT}s for $health_url"
    until health_request "$health_url"; do
        if (( SECONDS >= deadline )); then
            show_diagnostics
            exit 1
        fi
        sleep 2
    done

    success "CodeBuddy2API is healthy and running."
    printf '\nDashboard:  http://SERVER_IP:%s/\n' "$DEPLOY_PORT"
    printf 'Health:     http://SERVER_IP:%s/health\n' "$DEPLOY_PORT"
    printf 'Client mode: %s\n' "$CLIENT_AUTH_MODE"
    printf 'Environment: %s\n' "$ENV_FILE"

    if [[ "$ENV_CREATED" == "1" ]]; then
        printf '\n%bSave these generated credentials now:%b\n' "$YELLOW" "$RESET"
        printf 'CODEBUDDY_PASSWORD=%s\n' "$RELAY_PASSWORD"
        printf 'CODEBUDDY_ADMIN_PASSWORD=%s\n' "$ADMIN_PASSWORD"
        printf 'They are stored with permission 600 in %s.\n' "$ENV_FILE"
    fi

    printf '\nOperations:\n'
    printf '  Logs:    %s --env-file %q logs -f codebuddy2api\n' "${COMPOSE[*]}" "$ENV_FILE"
    printf '  Restart: %s --env-file %q restart codebuddy2api\n' "${COMPOSE[*]}" "$ENV_FILE"
    printf '  Stop:    %s --env-file %q down\n' "${COMPOSE[*]}" "$ENV_FILE"
}

main "$@"

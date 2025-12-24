#!/bin/bash
# =============================================================================
# JARVIS Unified Deployment Script v9.4
# =============================================================================
#
# Deploys the complete JARVIS stack to Docker (local) or Cloud Run (GCP).
#
# Features:
# - Multi-repo integration (JARVIS, JARVIS-Prime, Reactor-Core)
# - Docker Compose for local development
# - Terraform for Cloud Run production
# - Automatic image building and pushing
# - Health checks and verification
#
# Usage:
#   ./scripts/deploy.sh local                # Deploy locally with Docker Compose
#   ./scripts/deploy.sh local --all          # Deploy all services locally
#   ./scripts/deploy.sh cloud                # Deploy to Cloud Run
#   ./scripts/deploy.sh cloud --backend      # Deploy backend only
#   ./scripts/deploy.sh cloud --full         # Deploy backend + prime + redis
#   ./scripts/deploy.sh status               # Check deployment status
#   ./scripts/deploy.sh destroy              # Tear down deployment
#
# Version: 9.4.0
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/docker"
TERRAFORM_DIR="$PROJECT_ROOT/terraform"

# GCP Configuration
PROJECT_ID="${GCP_PROJECT_ID:-jarvis-473803}"
REGION="${GCP_REGION:-us-central1}"
JARVIS_VERSION="${JARVIS_VERSION:-9.4.0}"

# Repository paths
JARVIS_PRIME_PATH="${JARVIS_PRIME_PATH:-$PROJECT_ROOT/../jarvis-prime}"
REACTOR_CORE_PATH="${REACTOR_CORE_PATH:-$PROJECT_ROOT/../reactor-core}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# Logging Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[JARVIS]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[JARVIS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[JARVIS]${NC} WARNING: $1"
}

log_error() {
    echo -e "${RED}[JARVIS]${NC} ERROR: $1"
}

log_step() {
    echo -e "\n${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# =============================================================================
# Utility Functions
# =============================================================================

check_dependencies() {
    log_info "Checking dependencies..."

    local missing=()

    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing+=("docker")
    fi

    # Check Docker Compose
    if ! command -v docker &> /dev/null || ! docker compose version &> /dev/null; then
        missing+=("docker-compose")
    fi

    # Check gcloud for cloud deployment
    if [[ "$DEPLOY_TARGET" == "cloud" ]]; then
        if ! command -v gcloud &> /dev/null; then
            missing+=("gcloud")
        fi
        if ! command -v terraform &> /dev/null; then
            missing+=("terraform")
        fi
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing[*]}"
        exit 1
    fi

    log_success "All dependencies available"
}

check_env_file() {
    if [[ ! -f "$DOCKER_DIR/.env" ]]; then
        log_warning ".env file not found, creating from template..."
        if [[ -f "$DOCKER_DIR/.env.example" ]]; then
            cp "$DOCKER_DIR/.env.example" "$DOCKER_DIR/.env"
            log_info "Created .env from template. Please edit $DOCKER_DIR/.env with your settings."
        else
            log_error ".env.example not found"
            exit 1
        fi
    fi
}

check_repo_paths() {
    if [[ ! -d "$JARVIS_PRIME_PATH" ]]; then
        log_warning "JARVIS-Prime not found at $JARVIS_PRIME_PATH"
        log_info "Some features may be limited. Set JARVIS_PRIME_PATH to correct location."
    else
        log_success "JARVIS-Prime found: $JARVIS_PRIME_PATH"
    fi

    if [[ ! -d "$REACTOR_CORE_PATH" ]]; then
        log_warning "Reactor-Core not found at $REACTOR_CORE_PATH"
        log_info "Training features may be limited. Set REACTOR_CORE_PATH to correct location."
    else
        log_success "Reactor-Core found: $REACTOR_CORE_PATH"
    fi
}

# =============================================================================
# Docker Local Deployment
# =============================================================================

deploy_local() {
    local profile="${1:-default}"

    log_step "Deploying JARVIS Stack Locally (Profile: $profile)"

    cd "$DOCKER_DIR"
    check_env_file

    # Set environment for Docker Compose
    export JARVIS_VERSION="$JARVIS_VERSION"
    export JARVIS_PRIME_PATH="$JARVIS_PRIME_PATH"
    export REACTOR_CORE_PATH="$REACTOR_CORE_PATH"

    # Build images
    log_info "Building Docker images..."

    case "$profile" in
        "default")
            docker compose build jarvis-backend jarvis-frontend jarvis-training
            docker compose up -d
            ;;
        "local-llm")
            docker compose build jarvis-backend jarvis-frontend jarvis-training jarvis-prime
            docker compose --profile local-llm up -d
            ;;
        "full-training")
            docker compose build jarvis-backend jarvis-frontend jarvis-training reactor-core
            docker compose --profile full-training up -d
            ;;
        "all")
            docker compose build
            docker compose --profile all up -d
            ;;
        *)
            log_error "Unknown profile: $profile"
            exit 1
            ;;
    esac

    log_success "Local deployment complete!"

    # Show status
    echo ""
    docker compose ps

    # Health checks
    log_info "Waiting for services to become healthy..."
    sleep 10

    check_local_health
}

check_local_health() {
    log_info "Checking service health..."

    local services=("http://localhost:8010/health" "http://localhost:3000/health")
    local all_healthy=true

    for url in "${services[@]}"; do
        if curl -sf "$url" > /dev/null 2>&1; then
            log_success "$url - healthy"
        else
            log_warning "$url - not responding yet"
            all_healthy=false
        fi
    done

    if $all_healthy; then
        log_success "All services are healthy!"
    else
        log_info "Some services are still starting. Check with: docker compose logs -f"
    fi

    echo ""
    log_info "Access points:"
    echo "  - Backend API:  http://localhost:8010"
    echo "  - Frontend:     http://localhost:3000"
    echo "  - Loading Page: http://localhost:8011"
    echo "  - Redis:        localhost:6379"
    echo "  - ChromaDB:     http://localhost:8001"
}

stop_local() {
    log_step "Stopping Local Deployment"

    cd "$DOCKER_DIR"
    docker compose --profile all down

    log_success "Local deployment stopped"
}

destroy_local() {
    log_step "Destroying Local Deployment (including volumes)"

    cd "$DOCKER_DIR"
    docker compose --profile all down -v

    log_success "Local deployment destroyed"
}

# =============================================================================
# Cloud Run Deployment
# =============================================================================

deploy_cloud() {
    local mode="${1:-backend}"

    log_step "Deploying JARVIS to Cloud Run (Mode: $mode)"

    # Check GCP authentication
    log_info "Checking GCP authentication..."
    if ! gcloud auth print-identity-token &> /dev/null; then
        log_error "Not authenticated with GCP. Run: gcloud auth login"
        exit 1
    fi

    log_success "GCP authenticated as $(gcloud config get-value account)"

    # Configure Docker for Artifact Registry
    log_info "Configuring Docker for Artifact Registry..."
    gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

    # Build and push images
    case "$mode" in
        "backend")
            build_and_push_backend
            terraform_apply "enable_jarvis_backend=true"
            ;;
        "prime")
            build_and_push_prime
            terraform_apply "enable_jarvis_prime=true"
            ;;
        "full")
            build_and_push_backend
            build_and_push_prime
            terraform_apply "enable_jarvis_backend=true" "enable_jarvis_prime=true"
            ;;
        "with-redis")
            build_and_push_backend
            terraform_apply "enable_jarvis_backend=true" "enable_redis=true"
            ;;
        *)
            log_error "Unknown mode: $mode"
            exit 1
            ;;
    esac

    log_success "Cloud deployment complete!"
    show_cloud_status
}

build_and_push_backend() {
    log_info "Building JARVIS Backend image..."

    local image="$REGION-docker.pkg.dev/$PROJECT_ID/jarvis-backend/jarvis-backend:$JARVIS_VERSION"

    cd "$PROJECT_ROOT"
    docker build -f docker/Dockerfile.backend -t "$image" .
    docker tag "$image" "$REGION-docker.pkg.dev/$PROJECT_ID/jarvis-backend/jarvis-backend:latest"

    log_info "Pushing JARVIS Backend image..."
    docker push "$image"
    docker push "$REGION-docker.pkg.dev/$PROJECT_ID/jarvis-backend/jarvis-backend:latest"

    log_success "Backend image pushed: $image"
}

build_and_push_prime() {
    if [[ ! -d "$JARVIS_PRIME_PATH" ]]; then
        log_error "JARVIS-Prime not found at $JARVIS_PRIME_PATH"
        exit 1
    fi

    log_info "Building JARVIS-Prime image..."

    local image="$REGION-docker.pkg.dev/$PROJECT_ID/jarvis-prime/jarvis-prime:$JARVIS_VERSION"

    cd "$JARVIS_PRIME_PATH"
    docker build -t "$image" .
    docker tag "$image" "$REGION-docker.pkg.dev/$PROJECT_ID/jarvis-prime/jarvis-prime:latest"

    log_info "Pushing JARVIS-Prime image..."
    docker push "$image"
    docker push "$REGION-docker.pkg.dev/$PROJECT_ID/jarvis-prime/jarvis-prime:latest"

    log_success "Prime image pushed: $image"
}

terraform_apply() {
    log_info "Applying Terraform configuration..."

    cd "$TERRAFORM_DIR"

    # Initialize if needed
    if [[ ! -d ".terraform" ]]; then
        terraform init
    fi

    # Build var args
    local vars=""
    for var in "$@"; do
        vars="$vars -var=$var"
    done

    # Apply
    terraform apply $vars -auto-approve

    log_success "Terraform apply complete"
}

destroy_cloud() {
    log_step "Destroying Cloud Deployment"

    cd "$TERRAFORM_DIR"

    log_warning "This will destroy all Cloud Run services and infrastructure."
    read -p "Are you sure? (yes/no): " confirm

    if [[ "$confirm" == "yes" ]]; then
        terraform destroy -auto-approve
        log_success "Cloud deployment destroyed"
    else
        log_info "Destroy cancelled"
    fi
}

show_cloud_status() {
    log_step "Cloud Deployment Status"

    cd "$TERRAFORM_DIR"

    # Show Terraform outputs
    terraform output unified_stack_status 2>/dev/null || log_warning "No Terraform state found"

    echo ""
    log_info "Cloud Run Services:"
    gcloud run services list --platform managed --region "$REGION" --filter="metadata.labels.managed_by=terraform" 2>/dev/null || true
}

# =============================================================================
# Status and Info
# =============================================================================

show_status() {
    log_step "JARVIS Deployment Status"

    echo -e "${CYAN}Local Docker Status:${NC}"
    cd "$DOCKER_DIR" 2>/dev/null && docker compose ps 2>/dev/null || echo "  Not running"

    echo ""
    echo -e "${CYAN}Cloud Run Status:${NC}"
    gcloud run services list --platform managed --region "$REGION" 2>/dev/null || echo "  No services found"

    echo ""
    echo -e "${CYAN}Terraform State:${NC}"
    cd "$TERRAFORM_DIR" 2>/dev/null && terraform output unified_stack_status 2>/dev/null || echo "  No state"
}

show_help() {
    cat << EOF
${PURPLE}JARVIS Unified Deployment Script v9.4${NC}

${CYAN}Usage:${NC}
  ./scripts/deploy.sh <command> [options]

${CYAN}Commands:${NC}
  ${GREEN}local${NC} [profile]    Deploy locally with Docker Compose
                      Profiles: default, local-llm, full-training, all

  ${GREEN}cloud${NC} [mode]       Deploy to Google Cloud Run
                      Modes: backend, prime, full, with-redis

  ${GREEN}status${NC}             Show deployment status

  ${GREEN}stop${NC}               Stop local deployment

  ${GREEN}destroy${NC}            Destroy deployment (local or cloud)
                      --local  Destroy local Docker deployment
                      --cloud  Destroy Cloud Run deployment

  ${GREEN}help${NC}               Show this help message

${CYAN}Examples:${NC}
  ./scripts/deploy.sh local                 # Start default stack locally
  ./scripts/deploy.sh local all             # Start all services locally
  ./scripts/deploy.sh cloud backend         # Deploy backend to Cloud Run
  ./scripts/deploy.sh cloud full            # Deploy backend + prime
  ./scripts/deploy.sh status                # Check status
  ./scripts/deploy.sh destroy --local       # Stop and remove local

${CYAN}Environment Variables:${NC}
  GCP_PROJECT_ID        GCP project ID (default: jarvis-473803)
  GCP_REGION            GCP region (default: us-central1)
  JARVIS_VERSION        Docker image version (default: 9.4.0)
  JARVIS_PRIME_PATH     Path to jarvis-prime repo
  REACTOR_CORE_PATH     Path to reactor-core repo

${CYAN}Configuration:${NC}
  Docker:    docker/.env
  Terraform: terraform/terraform.tfvars

EOF
}

# =============================================================================
# Main Entry Point
# =============================================================================

main() {
    echo -e "${PURPLE}"
    echo "  ╔═══════════════════════════════════════════════════════════════╗"
    echo "  ║        JARVIS Unified Deployment Script v9.4                  ║"
    echo "  ║        Multi-Repo Docker + Cloud Run Integration              ║"
    echo "  ╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    local command="${1:-help}"
    shift || true

    case "$command" in
        "local")
            DEPLOY_TARGET="local"
            check_dependencies
            check_repo_paths
            deploy_local "${1:-default}"
            ;;
        "cloud")
            DEPLOY_TARGET="cloud"
            check_dependencies
            check_repo_paths
            deploy_cloud "${1:-backend}"
            ;;
        "status")
            show_status
            ;;
        "stop")
            stop_local
            ;;
        "destroy")
            case "${1:-}" in
                "--local")
                    destroy_local
                    ;;
                "--cloud")
                    destroy_cloud
                    ;;
                *)
                    log_error "Specify --local or --cloud"
                    exit 1
                    ;;
            esac
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"

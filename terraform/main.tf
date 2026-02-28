terraform {
  required_version = ">= 1.0.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "jarvis-473803-terraform-state"
    prefix = "prod"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# =============================================================================
# 💰 COST PROTECTION - Budget Alerts (FREE)
# =============================================================================
# ALWAYS enabled - protects you from unexpected charges.
# Set your billing_account_id to enable budget alerts.

module "budget" {
  source = "./modules/budget"

  project_id         = var.project_id
  billing_account_id = var.billing_account_id
  monthly_budget_usd = var.monthly_budget_usd
  alert_emails       = var.alert_emails
  alert_thresholds   = [0.25, 0.50, 0.75, 0.90, 1.0] # Alert at 25%, 50%, 75%, 90%, 100%
}

# =============================================================================
# 🌐 CORE INFRASTRUCTURE (Always enabled - $0 cost)
# =============================================================================
# These resources are completely FREE in GCP.

module "network" {
  source     = "./modules/network"
  project_id = var.project_id
  region     = var.region
}

module "security" {
  source     = "./modules/security"
  project_id = var.project_id
}

module "monitoring" {
  source             = "./modules/monitoring"
  project_id         = var.project_id
  monthly_budget_usd = var.monthly_budget_usd
}

# =============================================================================
# 🖥️ SPOT VM COMPUTE (Enabled by default - $0 until VMs run)
# =============================================================================
# Creates an instance TEMPLATE only (FREE). Actual VMs are created dynamically
# by gcp_vm_manager.py and cost ~$0.01-0.03/hour when running.
#
# Triple-Lock Safety ensures VMs auto-terminate after max_runtime.

module "compute" {
  count = var.enable_spot_vm_template ? 1 : 0

  source     = "./modules/compute"
  project_id = var.project_id
  region     = var.region
  zone       = var.zone
  network_id = module.network.vpc_id
  subnet_id  = module.network.subnet_id

  # Cost-optimized settings for solo developer
  machine_type        = var.spot_vm_machine_type
  disk_size_gb        = var.spot_vm_disk_size_gb
  max_runtime_seconds = var.spot_vm_max_runtime_hours * 3600

  depends_on = [module.network]
}

# =============================================================================
# 📦 REDIS / CLOUD MEMORYSTORE (DISABLED by default - saves ~$15/month)
# =============================================================================
# ⚠️ COST WARNING: Minimum ~$15/month even when idle!
#
# Only enable when you need:
# - Shared caching across multiple VMs
# - Persistent cache that survives VM restarts
# - Production deployment
#
# For development, use:
# - Local Redis: docker run -p 6379:6379 redis:alpine
# - In-memory caching (built into Ironcliw)

module "storage" {
  count = var.enable_redis ? 1 : 0

  source         = "./modules/storage"
  project_id     = var.project_id
  region         = var.region
  network_id     = module.network.vpc_id
  memory_size_gb = var.redis_memory_size_gb
  tier           = var.redis_tier

  depends_on = [module.network]
}

# =============================================================================
# 📊 COST ANALYSIS
# =============================================================================

locals {
  # Detailed cost breakdown
  cost_breakdown = {
    # Always-on resources
    network = {
      resource = "VPC + Firewall"
      cost     = "$0/month"
      notes    = "Completely free"
    }
    security = {
      resource = "Secret Manager"
      cost     = "$0/month"
      notes    = "Free tier: 10,000 operations/month"
    }
    monitoring = {
      resource = "Dashboards + Alerts"
      cost     = "$0/month"
      notes    = "Free tier: 100 alert policies"
    }
    budget = {
      resource = "Budget Alerts"
      cost     = "$0/month"
      notes    = "Always free"
    }

    # Optional resources
    spot_vm_template = {
      resource = "Instance Template"
      cost     = var.enable_spot_vm_template ? "$0/month (VMs: ~$0.01-0.03/hr when running)" : "Disabled"
      notes    = "Template is free, only pay for VM runtime"
    }
    redis = {
      resource = "Cloud Memorystore"
      cost     = var.enable_redis ? "~$${var.redis_memory_size_gb * 15}/month" : "Disabled ($0)"
      notes    = var.enable_redis ? "⚠️ This is the main cost driver" : "Using local/in-memory cache"
    }
  }

  # Total estimated monthly cost
  total_monthly_estimate = var.enable_redis ? var.redis_memory_size_gb * 15 : 0

  # Cost warnings
  cost_warnings = compact([
    var.enable_redis ? "⚠️ Redis enabled: ~$${var.redis_memory_size_gb * 15}/month" : "",
    var.spot_vm_max_runtime_hours > 3 ? "⚠️ VM max runtime > 3 hours increases orphan risk" : "",
    var.monthly_budget_usd > 20 ? "ℹ️ Budget set above $20/month" : "",
  ])
}

# =============================================================================
# 🧠 Ironcliw-PRIME TIER-0 BRAIN (DISABLED by default - Cloud Run)
# =============================================================================
# ⚠️ COST: Pay-per-request (~$0 when idle, ~$0.02-0.05/hr when running)
#
# Deploys Ironcliw-Prime to Cloud Run for serverless inference.
# Auto-scales 0-3 instances based on load.
#
# Prerequisites:
# 1. Build and push Docker image:
#    cd jarvis-prime
#    docker build -t us-central1-docker.pkg.dev/jarvis-473803/jarvis-prime/jarvis-prime:latest .
#    docker push us-central1-docker.pkg.dev/jarvis-473803/jarvis-prime/jarvis-prime:latest
#
# 2. Enable the module:
#    terraform apply -var="enable_jarvis_prime=true"

module "jarvis_prime" {
  count  = var.enable_jarvis_prime ? 1 : 0
  source = "./modules/jarvis_prime"

  project_id  = var.project_id
  region      = var.region
  environment = var.developer_mode ? "dev" : "prod"

  # Cloud Run configuration
  image_tag      = var.jarvis_prime_image_tag
  min_instances  = var.jarvis_prime_min_instances
  max_instances  = var.jarvis_prime_max_instances
  memory         = var.jarvis_prime_memory
  cpu            = var.jarvis_prime_cpu
  model_gcs_path = var.jarvis_prime_model_gcs_path

  # Optional: Connect to Redis (if enabled)
  redis_host = var.enable_redis && length(module.storage) > 0 ? module.storage[0].redis_host : ""
  redis_port = var.enable_redis && length(module.storage) > 0 ? module.storage[0].redis_port : 6379

  # Optional: VPC access for Redis
  network_id = var.enable_redis ? module.network.vpc_id : null
  subnet_id  = var.enable_redis ? module.network.subnet_id : null

  depends_on = [module.network]
}

# =============================================================================
# 🤖 Ironcliw BACKEND CLOUD RUN (v9.4)
# =============================================================================
# ⚠️ COST: Pay-per-request (~$0 when idle, ~$0.05-0.15/hr when running)
#
# Deploys the full Ironcliw-AI-Agent backend with:
# - Neural Mesh (60+ agents)
# - Data Flywheel
# - Intelligent Continuous Scraping
# - Multi-repo integration
#
# Prerequisites:
# 1. Build and push Docker image:
#    docker build -f docker/Dockerfile.backend -t us-central1-docker.pkg.dev/jarvis-473803/jarvis-backend/jarvis-backend:latest .
#    docker push us-central1-docker.pkg.dev/jarvis-473803/jarvis-backend/jarvis-backend:latest
#
# 2. Set ANTHROPIC_API_KEY in Secret Manager:
#    gcloud secrets create anthropic-api-key --replication-policy=automatic
#    echo -n "sk-ant-..." | gcloud secrets versions add anthropic-api-key --data-file=-
#
# 3. Enable the module:
#    terraform apply -var="enable_jarvis_backend=true"

module "jarvis_backend" {
  count  = var.enable_jarvis_backend ? 1 : 0
  source = "./modules/jarvis_backend"

  project_id  = var.project_id
  region      = var.region
  environment = var.developer_mode ? "dev" : "prod"

  # Cloud Run configuration
  image_tag     = var.jarvis_backend_image_tag
  min_instances = var.jarvis_backend_min_instances
  max_instances = var.jarvis_backend_max_instances
  memory        = var.jarvis_backend_memory
  cpu           = var.jarvis_backend_cpu
  concurrency   = var.jarvis_backend_concurrency

  # Neural Mesh configuration
  neural_mesh_enabled    = var.neural_mesh_enabled
  neural_mesh_max_agents = var.neural_mesh_max_agents

  # Integration with Ironcliw-Prime (if enabled)
  jarvis_prime_url = var.enable_jarvis_prime && length(module.jarvis_prime) > 0 ? module.jarvis_prime[0].service_url : ""

  # Optional: Connect to Redis (if enabled)
  redis_host = var.enable_redis && length(module.storage) > 0 ? module.storage[0].redis_host : ""
  redis_port = var.enable_redis && length(module.storage) > 0 ? module.storage[0].redis_port : 6379

  # Optional: GCS bucket for models
  gcs_bucket = var.jarvis_gcs_bucket

  # Secret Manager
  anthropic_api_key_secret = var.anthropic_api_key_secret

  # Optional: VPC access for Redis
  network_id = var.enable_redis ? module.network.vpc_id : null
  subnet_id  = var.enable_redis ? module.network.subnet_id : null

  depends_on = [module.network, module.jarvis_prime]
}


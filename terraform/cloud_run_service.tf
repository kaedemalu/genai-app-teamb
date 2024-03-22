resource "google_cloud_run_service" "app" {
  name     = "genai-app-v2"
  location = "us-central1"

  template {
    spec {
      containers {
        image = "asia-northeast1-docker.pkg.dev/${var.project}/genai-repo/genai_app:latest"
        env {
          name  = "PROJECT_ID"
          value = var.project
        }
        env {
          name  = "REGION"
          value = "us-central1"
        }
        ports {
          container_port = 8080
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = false
  }

  lifecycle {
    ignore_changes = [traffic[0].revision_name]
  }
}

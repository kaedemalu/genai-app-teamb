resource "google_artifact_registry_repository" "app" {
  location      = var.region
  repository_id = "genai-repo"
  format        = "DOCKER"
}

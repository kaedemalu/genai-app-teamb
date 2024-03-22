variable "project" {
  type        = string
  description = "String of Project ID"
}

variable "region" {
  type        = string
  description = "String of Service Region"
}

variable "slack_bot_token" {
  type        = string
  description = "String of Slack Bot Token"
  validation {
    condition     = can(regex("^xoxb-*", var.slack_bot_token))
    error_message = "slack_bot_token should start xoxb-"
  }
}

variable "slack_signing_secret" {
  type        = string
  description = "String of Slack Signing Secret"
}

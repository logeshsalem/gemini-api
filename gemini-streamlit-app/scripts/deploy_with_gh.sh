#!/usr/bin/env bash
set -euo pipefail

REPO_NAME=${1:-gemini-streamlit-app}

if ! command -v gh &> /dev/null; then
  echo "GitHub CLI (gh) is not installed. Install from https://cli.github.com/" >&2
  exit 1
fi

echo "Creating repo $REPO_NAME and pushing…"

gh repo create "$REPO_NAME" --public --source=. --remote=origin --push

echo "Done! Visit: https://github.com/$(gh api user | jq -r .login)/$REPO_NAME"

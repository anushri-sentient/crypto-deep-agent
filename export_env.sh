#!/bin/bash
# Usage: source export_env.sh
# This script will export all variables from .env into your current shell session.
# It ignores comments and blank lines, and handles simple KEY=VALUE pairs.

if [ ! -f .env ]; then
  echo ".env file not found in current directory."
  return 1
fi

export $(grep -v '^#' .env | grep -v '^$' | xargs)
echo "All variables from .env have been exported to your shell session." 
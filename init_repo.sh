#!/bin/bash
# Usage: bash init_repo.sh <YOUR_GITHUB_TOKEN>
set -e

TOKEN="$1"
if [ -z "$TOKEN" ]; then
  echo "Usage: bash init_repo.sh <GITHUB_TOKEN>"
  exit 1
fi

REPO_NAME="PoreTracker-OF"
GITHUB_USER=$(curl -s -H "Authorization: token $TOKEN" https://api.github.com/user | python3 -c "import sys,json; print(json.load(sys.stdin)['login'])")

echo "Creating repo as: $GITHUB_USER"

# Create GitHub repo
curl -s -X POST \
  -H "Authorization: token $TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/user/repos \
  -d "{\"name\":\"$REPO_NAME\",\"description\":\"Pore-scale tracking and analysis tools built on OpenFOAM\",\"private\":false,\"auto_init\":false}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('Repo URL:', d.get('html_url', d.get('message','failed')))"

# Init git locally
cd "$(dirname "$0")"
git init
git add .
git commit -m "Initial commit: project structure and README"
git branch -M main
git remote add origin "https://$GITHUB_USER:$TOKEN@github.com/$GITHUB_USER/$REPO_NAME.git"
git push -u origin main

echo ""
echo "Done! Visit: https://github.com/$GITHUB_USER/$REPO_NAME"

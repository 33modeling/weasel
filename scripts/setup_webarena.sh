#!/usr/bin/env bash
# Optional + HEAVY: stand up the self-hosted WebArena websites and emit the
# WA_* environment file that scripts/run_eval.sh --bench webarena consumes.
#
# WebArena is NOT pip-installable infrastructure — the sites ship as large
# Docker image tarballs (shopping/Magento, shopping_admin, gitlab, forum,
# wikipedia .zim, openstreetmap). Two paths:
#
#   [RECOMMENDED] Official AWS AMI with everything preloaded:
#       ami-08a862bf98e3bd7aa (us-east-2), t3a.xlarge + 1000GB EBS.
#       Boot it, note its public DNS, then run this script with HOST=<dns>
#       to just WRITE the env file (no local docker needed):
#         HOST=ec2-xx.compute.amazonaws.com bash scripts/setup_webarena.sh env
#
#   [MANUAL] Load the image tarballs on THIS 8xA100 box (needs docker + ~1TB):
#         bash scripts/setup_webarena.sh up      # docker run the 6 sites
#
# See https://github.com/web-arena-x/webarena (Environment setup) for the
# tarball download links and reset/auth steps. WebArena-Lite (165 tasks) only
# needs shopping, shopping_admin, gitlab, forum, homepage (skip wikipedia/map).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${WEASEL_WORK:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."; source scripts/setup_env.sh
fi

ACTION="${1:-env}"
HOST="${HOST:-localhost}"
ENV_FILE="$WEASEL_WORK/webarena_env.sh"

# Canonical WebArena ports (official defaults).
P_SHOP=7770; P_SHOP_ADMIN=7780; P_FORUM=9999; P_GITLAB=8023; P_WIKI=8888; P_MAP=3000; P_HOME=4399

write_env() {
  cat > "$ENV_FILE" <<EOF
# WebArena site URLs — source before scripts/run_eval.sh --bench webarena
# Generated for HOST=$HOST. Edit if your ports/host differ.
export WA_SHOPPING="http://$HOST:$P_SHOP"
export WA_SHOPPING_ADMIN="http://$HOST:$P_SHOP_ADMIN/admin"
export WA_REDDIT="http://$HOST:$P_FORUM"
export WA_GITLAB="http://$HOST:$P_GITLAB"
export WA_WIKIPEDIA="http://$HOST:$P_WIKI/wikipedia_en_all_maxi_20220215/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="http://$HOST:$P_MAP"
export WA_HOMEPAGE="http://$HOST:$P_HOME"
# BrowserGym reads these same WA_* names.
EOF
  echo "[setup_webarena] wrote $ENV_FILE"
  echo "[setup_webarena] before eval:  source $ENV_FILE"
}

case "$ACTION" in
  env)
    write_env
    ;;
  up)
    command -v docker >/dev/null || { echo "[error] docker not installed on this VM." >&2; exit 1; }
    echo "[setup_webarena] NOTE: this assumes the WebArena image tarballs are already"
    echo "                 'docker load'-ed (shopping, shopping_admin, gitlab, forum, etc.)."
    echo "                 Download links + exact image names: github.com/web-arena-x/webarena"
    # Standard launches (image names follow the WebArena distribution; adjust to your loaded tags):
    docker run -d --name shopping       -p ${P_SHOP}:80        shopping_final_0712       || true
    docker run -d --name shopping_admin -p ${P_SHOP_ADMIN}:80  shopping_admin_final_0719 || true
    docker run -d --name gitlab         -p ${P_GITLAB}:8023    gitlab-populated-final-port8023 || true
    docker run -d --name forum          -p ${P_FORUM}:80       postmill-populated-exposed-withimg || true
    echo "[setup_webarena] containers launched (verify: docker ps). Wait ~1-2 min for boot."
    write_env
    echo "[setup_webarena] Reminder: run WebArena's reset + auth-cookie regen before each eval."
    ;;
  *) echo "usage: bash scripts/setup_webarena.sh {env|up}   (HOST=<dns> for env)" >&2; exit 2 ;;
esac

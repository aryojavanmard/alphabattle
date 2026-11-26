#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-pokemon-showdown}"

NODE_MAJOR=$(node -v | sed 's/^v//' | cut -d. -f1)
if [ "$NODE_MAJOR" -lt 22 ]; then
  echo "Error: Pokémon Showdown requires Node.js 22+ (found $(node -v))." >&2
  exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
  git clone --depth 1 https://github.com/smogon/pokemon-showdown.git "$TARGET_DIR"
fi

cd "$TARGET_DIR"
npm install
if [ ! -f config/config.js ]; then
  cp config/config-example.js config/config.js
fi

node -e '
const fs = require("fs");
let cfg = fs.readFileSync("config/config.js", "utf8");
cfg = cfg.replace(/exports\.noguestsecurity\s*=\s*false;/, "exports.noguestsecurity = true;");
cfg = cfg.replace(/exports\.nothrottle\s*=\s*false;/, "exports.nothrottle = true;");
cfg = cfg.replace(/exports\.noipchecks\s*=\s*false;/, "exports.noipchecks = true;");
fs.writeFileSync("config/config.js", cfg, "utf8");
'

node build
echo "Ready! Start with: (cd $TARGET_DIR && node pokemon-showdown start --no-security --skip-build 8000)"

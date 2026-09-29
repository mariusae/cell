#!/bin/sh
# Sync the working tree to a sprite and restart the demo service there.
#
#   demo/deploy.sh            # the "cell" sprite
#   SPRITE=other demo/deploy.sh
#
# Sends every file git knows about or would add (uncommitted changes
# included, ignored files excluded), replaces the copy in ~/cell on the
# sprite (keeping its .venv), runs uv sync, and (re)creates the "demo"
# service with --http-port 8080 so requests wake the sprite and start it.
set -eu

SPRITE=${SPRITE:-cell}
cd "$(git rev-parse --show-toplevel)"

git ls-files -co --exclude-standard -z |
	COPYFILE_DISABLE=1 tar czf - --no-mac-metadata --null -T - |
	sprite exec -s "$SPRITE" -- sh -c '
		set -e
		mkdir -p "$HOME/cell" && cd "$HOME/cell"
		find . -mindepth 1 -maxdepth 1 ! -name .venv -exec rm -rf {} +
		tar xzf - 2>/dev/null
		uv sync -q
		if sprite-env services get demo >/dev/null 2>&1; then
			sprite-env services restart demo --duration 3s >/dev/null
		else
			sprite-env services create demo --cmd "$(command -v uv)" \
				--args run,python,-m,demo --env HOST=0.0.0.0,PORT=8080 \
				--dir "$HOME/cell" --http-port 8080 --duration 3s >/dev/null
		fi
		sleep 2
		curl -fsS -o /dev/null localhost:8080/api/scenarios && echo "demo: running"
	'

sprite url -s "$SPRITE" 2>/dev/null | sed -n "s/^URL: //p"

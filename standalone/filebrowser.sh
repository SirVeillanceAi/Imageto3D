#!/usr/bin/env bash
# File manager for the pod: File Browser (https://filebrowser.org) serving /workspace on port 8080.
# Sourced by start.sh; expects log/warn, BASE_DIR, WORKSPACE and LOG_DIR to be set.
#
# Login: FILEBROWSER_USERNAME / FILEBROWSER_PASSWORD, else GRADIO_USERNAME / GRADIO_PASSWORD,
# else user "admin" with a random password written to /workspace/.filebrowser/login.txt
# (and printed once in the pod logs).

start_filebrowser() {
    [ "${FILEBROWSER:-1}" = "1" ] || { log "File manager disabled (FILEBROWSER=0)"; return 0; }

    local ver="${FILEBROWSER_VERSION:-v2.63.23}"
    local port="${FILEBROWSER_PORT:-8080}"
    local bin="$BASE_DIR/bin/filebrowser"
    local dir="$WORKSPACE/.filebrowser"
    local db="$dir/filebrowser.db"
    local login="$dir/login.txt"
    mkdir -p "$BASE_DIR/bin" "$dir"
    chmod 700 "$dir"

    if [ ! -x "$bin" ] || ! "$bin" version 2>/dev/null | grep -q "${ver#v}"; then
        log "Installing File Browser $ver"
        local tmp
        tmp="$(mktemp -d)"
        if curl -fsSL "https://github.com/filebrowser/filebrowser/releases/download/$ver/linux-amd64-filebrowser.tar.gz" \
                -o "$tmp/fb.tgz" && tar -xzf "$tmp/fb.tgz" -C "$tmp" filebrowser; then
            install -m 755 "$tmp/filebrowser" "$bin"
            rm -rf "$tmp"
        else
            rm -rf "$tmp"
            warn "File Browser download failed - file manager disabled this boot."
            return 0
        fi
    fi

    local user="${FILEBROWSER_USERNAME:-${GRADIO_USERNAME:-admin}}"
    local pass="${FILEBROWSER_PASSWORD:-${GRADIO_PASSWORD:-}}"
    if [ -n "$pass" ] && [ "${#pass}" -lt 8 ]; then
        warn "File Browser passwords need 8+ characters - ignoring the one you set and generating one instead."
        pass=""
    fi

    if [ ! -f "$db" ]; then
        local generated=0
        if [ -z "$pass" ]; then
            pass="$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 20)"
            generated=1
        fi
        "$bin" -d "$db" config init --minimumPasswordLength 8 >/dev/null \
            && "$bin" -d "$db" users add "$user" "$pass" --perm.admin >/dev/null \
            || { warn "File Browser setup failed - file manager disabled."; rm -f "$db"; return 0; }
        if [ "$generated" = 1 ]; then
            printf 'File Browser login\nuser: %s\npassword: %s\n' "$user" "$pass" > "$login"
            log "File Browser login  ->  user: $user   password: $pass   (saved in $login)"
        else
            printf 'File Browser login\nuser: %s\npassword: same as FILEBROWSER_PASSWORD / GRADIO_PASSWORD\n' "$user" > "$login"
        fi
        chmod 600 "$login"
    elif [ -n "$pass" ]; then
        # keep the login in step with the pod's environment variables
        "$bin" -d "$db" users update "$user" -p "$pass" --perm.admin >/dev/null 2>&1 \
            || "$bin" -d "$db" users add "$user" "$pass" --perm.admin >/dev/null 2>&1 \
            || warn "Could not update the File Browser login."
        if ! grep -qx "user: $user" "$login" 2>/dev/null; then
            printf '\nuser: %s\npassword: same as FILEBROWSER_PASSWORD / GRADIO_PASSWORD\n' "$user" >> "$login"
            chmod 600 "$login"
        fi
    fi

    "$bin" -d "$db" config set --address 0.0.0.0 --port "$port" --root "$WORKSPACE" >/dev/null \
        || warn "Could not apply File Browser settings."

    log "File manager on port $port  ->  https://${RUNPOD_POD_ID:-<pod-id>}-${port}.proxy.runpod.net  (login: $login)"
    (
        while true; do
            "$bin" -d "$db" >> "$LOG_DIR/filebrowser.log" 2>&1
            echo "[imageto3d] File Browser exited; restarting in 5 s" >> "$LOG_DIR/filebrowser.log"
            sleep 5
        done
    ) &
}

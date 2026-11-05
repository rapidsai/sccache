# syntax=docker/dockerfile:1

ARG BASE
FROM $BASE

SHELL ["/bin/bash", "-ceuxo", "pipefail"]

ARG RUST_VERSION=1.91.0
ARG CARGO_BUILD_TARGET
ARG LOCAL_WORKSPACE_FOLDER

RUN <<EORUN
if grep ubuntu:x:1000:1000 /etc/passwd >/dev/null; then userdel -f -r ubuntu; fi

mkdir -p /opt "${LOCAL_WORKSPACE_FOLDER}"
ln -s "${LOCAL_WORKSPACE_FOLDER}" /opt/sccache

apt update
apt install -y pkg-config bubblewrap libcap-dev

rustup override set "$RUST_VERSION"
rustup target add "$CARGO_BUILD_TARGET"
rustup component add rustfmt clippy llvm-tools-preview
rustup completions bash > /usr/share/bash-completion/completions/rustup

# Link the musl target through the musl-cross gcc driver but with the rust-lld that ships
# with the toolchain instead of GNU ld, which spends ~13s per debug binary. rustc puts
# <sysroot>/lib/rustlib/<host>/bin first on the linker's PATH, so this always picks the
# rust-lld of the toolchain doing the build.
cat <<"EOF" > /usr/local/bin/musl-gcc-rust-lld
#!/bin/sh
lld="$(command -v rust-lld)" && exec "${TARGET_CC:?}" -fuse-ld=lld "-B${lld%/*}/gcc-ld" "$@"
exec "${TARGET_CC:?}" "$@"
EOF
chmod +x /usr/local/bin/musl-gcc-rust-lld
sed -i "/^\[target\.$CARGO_BUILD_TARGET\]/,/^\[/ s|^linker = .*|linker = \"/usr/local/bin/musl-gcc-rust-lld\"|" "$CARGO_HOME/config.toml"

sed -i 's/HISTSIZE=.*/HISTSIZE=/g' /etc/skel/.bashrc
sed -i 's/HISTFILESIZE=.*/HISTFILESIZE=/g' /etc/skel/.bashrc

cat <<"EOF" | tee -a /etc/skel/.bashrc

HISTFILE=/opt/sccache/.devcontainer/._bash_history
XDG_CACHE_HOME=/opt/sccache/.devcontainer/.cache
XDG_CONFIG_HOME=/opt/sccache/.devcontainer/.config

export PROMPT_COMMAND="${PROMPT_COMMAND:-history -a}"
if test -n "${PROMPT_COMMAND##*"history -a"*}"; then
    export PROMPT_COMMAND="history -a; $PROMPT_COMMAND"
fi
if test -d /opt/sccache/.devcontainer/bin; then
    if test -n "${PATH##*"/opt/sccache/.devcontainer/bin"*}"; then
        export PATH="/opt/sccache/.devcontainer/bin:$PATH"
    fi
fi
EOF

EORUN

ENV CARGO_BUILD_TARGET="$CARGO_BUILD_TARGET"
ENV RUST_VERSION="$RUST_VERSION"
ENV SCCACHE_REPO="/opt/sccache"

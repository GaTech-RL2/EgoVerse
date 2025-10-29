# sudo mkdir -p /usr/local/nvm
# sudo git clone --depth=1 https://github.com/nvm-sh/nvm.git /usr/local/nvm
# sudo chmod -R a+rX /usr/local/nvm


# /etc/profile.d/nvm.sh
cat <<'EOF' | sudo tee /etc/profile.d/nvm.sh >/dev/null
export NVM_DIR="/usr/local/nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && . "$NVM_DIR/bash_completion"
EOF
sudo chmod 644 /etc/profile.d/nvm.sh

# also cover interactive non-login shells
echo '[ -s /etc/profile.d/nvm.sh ] && . /etc/profile.d/nvm.sh' \
| sudo tee -a /etc/bash.bashrc >/dev/null
echo '[ -s /etc/profile.d/nvm.sh ] && . /etc/profile.d/nvm.sh' \
| sudo tee -a /etc/zsh/zshrc    >/dev/null 2>/dev/null || true


# really important fix here
# some edits to setup_nvm

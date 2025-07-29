#!/bin/bash
set -e

echo "=== macOS用 Docker + Colima インストールスクリプト ==="
echo ""
echo "📦 Docker CLI + Docker Compose + Colima をインストールします。"
echo "⚠️ Docker Desktop は使用しません。"
echo ""

# macOSチェック
if [[ "$OSTYPE" != "darwin"* ]]; then
  echo "❌ このスクリプトは macOS 専用です。他のOSでは実行しないでください。"
  exit 1
fi

# Homebrewのインストール（存在しなければ）
if ! command -v brew >/dev/null 2>&1; then
  echo "🔍 Homebrew が見つかりません。インストールを開始します..."
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
  eval "$(/opt/homebrew/bin/brew shellenv)"
  echo "✅ Homebrew のインストールが完了しました。"
else
  echo "✅ Homebrew はすでにインストールされています。"
fi

# Docker + Colima のインストール
echo ""
echo "🔧 docker, docker-compose, colima をインストールします..."
brew install docker docker-compose colima

# Colima の初期起動
echo ""
echo "🚀 Colima を起動しています（初回は仮想マシンを作成します）..."
colima start

# 確認メッセージ
echo ""
echo "✅ Colima + Docker のインストールが完了しました！"
echo ""
echo "📋 次のステップ:"
echo "1. ターミナルで 'docker info' を実行して動作確認"
echo "2. 必要に応じて 'docker compose up -d' でコンテナ起動"
echo ""
echo "🔧 操作コマンド例:"
echo "  - Colima 再起動: colima restart"
echo "  - Colima 停止: colima stop"
echo "  - Docker 起動確認: docker info"
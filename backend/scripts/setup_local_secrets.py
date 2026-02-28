#!/usr/bin/env python3
"""
Setup Local Secrets for Ironcliw Development
Stores secrets securely in macOS Keychain for local development
"""

import getpass
import sys
import keyring

SECRETS = {
    "anthropic-api-key": {
        "description": "Anthropic API Key",
        "example": "sk-ant-api03-...",
        "help": "Get from https://console.anthropic.com/settings/keys",
    },
    "jarvis-db-password": {
        "description": "Database Password",
        "example": "Your secure password",
        "help": "Password for Ironcliw PostgreSQL database",
    },
    "picovoice-access-key": {
        "description": "Picovoice Access Key",
        "example": "xxxxx==",
        "help": "Get from https://console.picovoice.ai",
    },
}


def setup_keychain_secrets():
    """Store secrets in macOS Keychain"""
    print("=" * 60)
    print("🔐 Ironcliw Local Secret Setup")
    print("=" * 60)
    print()
    print("This will store secrets securely in your macOS Keychain.")
    print("These secrets will be used for local development only.\n")

    for secret_id, config in SECRETS.items():
        print("─" * 60)
        print(f"\n📝 {config['description']}")
        print(f"   Secret ID: {secret_id}")
        print(f"   Example: {config['example']}")
        print(f"   Help: {config['help']}\n")

        # Check if already exists
        existing = keyring.get_password("Ironcliw", secret_id)
        if existing:
            print(f"   ✓ Secret already exists (hidden)")
            update = input("   Update? (y/N): ").lower().strip()
            if update != 'y':
                print("   ⏭️  Skipped\n")
                continue

        # Get new value
        secret_value = getpass.getpass(f"   Enter {config['description']}: ")

        if not secret_value or secret_value.strip() == "":
            print("   ⚠️  Empty value, skipping\n")
            continue

        # Store in keychain
        try:
            keyring.set_password("Ironcliw", secret_id, secret_value.strip())
            print("   ✅ Stored securely in Keychain\n")
        except Exception as e:
            print(f"   ❌ Failed to store in keychain: {e}\n")
            continue

    print("─" * 60)
    print("\n✅ Secret configuration complete!")
    print("\n🔍 Your secrets are now stored in macOS Keychain:")
    print("   • View: Keychain Access app → search for 'Ironcliw'")
    print("   • Delete: Run this script with --clear flag")
    print("   • Update: Run this script again\n")

    print("📚 Next steps:")
    print("   1. Test: python backend/core/secret_manager.py")
    print("   2. Start Ironcliw and it will automatically use Keychain secrets\n")


def clear_keychain_secrets():
    """Remove all Ironcliw secrets from Keychain"""
    print("=" * 60)
    print("🗑️  Clear Ironcliw Secrets from Keychain")
    print("=" * 60)
    print()

    confirm = input("⚠️  Are you sure you want to delete all Ironcliw secrets? (yes/N): ")
    if confirm.lower() != "yes":
        print("❌ Cancelled")
        return

    print()
    for secret_id, config in SECRETS.items():
        try:
            existing = keyring.get_password("Ironcliw", secret_id)
            if existing:
                keyring.delete_password("Ironcliw", secret_id)
                print(f"✅ Deleted: {config['description']}")
            else:
                print(f"⏭️  Not found: {config['description']}")
        except Exception as e:
            print(f"❌ Failed to delete {secret_id}: {e}")

    print("\n✅ Cleanup complete!")


def list_keychain_secrets():
    """List which secrets are configured in Keychain"""
    print("=" * 60)
    print("📋 Ironcliw Secrets Status")
    print("=" * 60)
    print()

    for secret_id, config in SECRETS.items():
        try:
            existing = keyring.get_password("Ironcliw", secret_id)
            if existing:
                # Show partial value for verification
                if len(existing) > 20:
                    preview = f"{existing[:10]}...{existing[-5:]}"
                else:
                    preview = "*" * len(existing)
                print(f"✅ {config['description']:30} {preview}")
            else:
                print(f"❌ {config['description']:30} Not set")
        except Exception as e:
            print(f"⚠️  {config['description']:30} Error: {e}")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage Ironcliw secrets in macOS Keychain"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all Ironcliw secrets from Keychain",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List status of all Ironcliw secrets",
    )

    args = parser.parse_args()

    if args.clear:
        clear_keychain_secrets()
    elif args.list:
        list_keychain_secrets()
    else:
        setup_keychain_secrets()

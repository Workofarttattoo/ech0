#!/usr/bin/env python3
"""
HuggingFace Authentication Setup Helper
Helps users authenticate with HuggingFace Hub for model access
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def check_hf_installation():
    """Check if huggingface_hub is installed"""
    try:
        from huggingface_hub import login, whoami, HfApi
        return True
    except ImportError:
        logger.error("❌ huggingface_hub is not installed")
        logger.error("Install it with: pip install huggingface-hub")
        return False


def check_existing_token():
    """Check if HuggingFace token already exists"""
    hf_home = os.path.expanduser("~/.huggingface")
    token_file = os.path.join(hf_home, "token")

    if os.path.exists(token_file):
        logger.info("✅ Found existing HuggingFace token")
        return True

    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        logger.info("✅ Found HF_TOKEN environment variable")
        return True

    return False


def verify_token():
    """Verify that the HuggingFace token is valid"""
    try:
        from huggingface_hub import whoami

        try:
            user_info = whoami()
            logger.info(f"✅ Token is valid! Logged in as: {user_info['name']}")
            return True
        except Exception as e:
            if "401" in str(e) or "unauthorized" in str(e).lower():
                logger.error("❌ Your HuggingFace token is invalid or expired")
                return False
            else:
                logger.error(f"Error verifying token: {e}")
                return False
    except ImportError:
        logger.error("huggingface_hub not installed")
        return False


def login_to_huggingface():
    """Authenticate with HuggingFace"""
    try:
        from huggingface_hub import login

        logger.info("\n📝 Authenticating with HuggingFace Hub...")
        logger.info("You'll be prompted to enter your HuggingFace token.")
        logger.info("Get your token from: https://huggingface.co/settings/tokens")
        logger.info("\nSteps:")
        logger.info("1. Visit https://huggingface.co/settings/tokens")
        logger.info("2. Create a new token (with 'repo' read access)")
        logger.info("3. Paste the token below\n")

        login(token=None, add_to_git_credential_store=True)
        logger.info("✅ Authentication successful!")
        return True
    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        return False


def setup_environment_variable():
    """Help user set HF_TOKEN environment variable"""
    logger.info("\n🔧 Setting up HF_TOKEN environment variable...")

    token = input("Enter your HuggingFace token: ").strip()

    if not token:
        logger.error("No token provided")
        return False

    # Add to .bashrc, .zshrc, or equivalent
    shell_config = None
    if os.path.exists(os.path.expanduser("~/.bashrc")):
        shell_config = os.path.expanduser("~/.bashrc")
    elif os.path.exists(os.path.expanduser("~/.zshrc")):
        shell_config = os.path.expanduser("~/.zshrc")

    if shell_config:
        # Check if already exists
        with open(shell_config, 'r') as f:
            content = f.read()
            if 'HF_TOKEN' in content:
                logger.warning("⚠️  HF_TOKEN already exists in shell config")
                response = input("Update it? (y/n): ").strip().lower()
                if response != 'y':
                    return False

        # Add to shell config
        with open(shell_config, 'a') as f:
            f.write(f'\nexport HF_TOKEN="{token}"\n')
        logger.info(f"✅ Added HF_TOKEN to {shell_config}")
        logger.info("⚠️  Please run: source ~/.bashrc (or ~/.zshrc)")
        return True
    else:
        logger.warning("Could not find shell config file (.bashrc or .zshrc)")
        logger.info(f"Manually set: export HF_TOKEN=\"{token}\"")
        return False


def main():
    """Main setup flow"""
    logger.info("=" * 60)
    logger.info("🔐 HuggingFace Authentication Setup")
    logger.info("=" * 60)

    # Check if huggingface_hub is installed
    if not check_hf_installation():
        sys.exit(1)

    # Check existing token
    if check_existing_token():
        logger.info("\nVerifying your token...")
        if verify_token():
            logger.info("✅ You're all set! Your token is valid.")
            sys.exit(0)
        else:
            logger.info("\nYour token is invalid or expired. Let's get a new one.\n")
    else:
        logger.info("\nNo HuggingFace token found. Let's set one up.\n")

    # Ask user preference
    logger.info("Choose authentication method:")
    logger.info("1. Interactive login (recommended)")
    logger.info("2. Set HF_TOKEN environment variable")

    choice = input("\nEnter choice (1 or 2): ").strip()

    success = False
    if choice == "1":
        success = login_to_huggingface()
    elif choice == "2":
        success = setup_environment_variable()
    else:
        logger.error("Invalid choice")
        sys.exit(1)

    if success:
        logger.info("\n✅ Setup complete!")
        logger.info("\nYou can now run:")
        logger.info("  python ech0_train_orchestrator.py")
        sys.exit(0)
    else:
        logger.error("\n❌ Setup failed. Please try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()

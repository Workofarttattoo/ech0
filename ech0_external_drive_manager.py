#!/usr/bin/env python3
"""
ech0 External Drive Manager
Automatically detect and use external drives for wisdom ingestion/training data storage
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExternalDriveManager:
    """Manages external drive detection and path configuration for ech0 training data"""

    def __init__(self, preferred_label: Optional[str] = None):
        """
        Initialize external drive manager

        Args:
            preferred_label: Preferred drive label to look for (e.g., "ECH0_DRIVE")
        """
        self.preferred_label = preferred_label
        self.default_local_path = Path("./ech0_training_data")

    def detect_external_drives(self) -> List[Tuple[str, str, str]]:
        """
        Detect all mounted external drives

        Returns:
            List of tuples: (mount_point, device, filesystem_type)
        """
        external_drives = []

        try:
            # Try using lsblk if available
            result = subprocess.run(
                ['lsblk', '-o', 'NAME,MOUNTPOINT,FSTYPE,TYPE', '-J'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                for device in data.get('blockdevices', []):
                    self._extract_external_mounts(device, external_drives)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            # Fallback to checking common mount points (macOS doesn't have lsblk)
            logger.debug(f"lsblk not available ({e.__class__.__name__}), using fallback detection")
            external_drives = self._detect_via_mount_points()

        return external_drives

    def _extract_external_mounts(self, device: dict, external_drives: List[Tuple[str, str, str]]):
        """Recursively extract external drive mount points from lsblk output"""
        mountpoint = device.get('mountpoint')
        fstype = device.get('fstype', '')
        name = device.get('name', '')
        device_type = device.get('type', '')

        # Check if this is a mounted partition/disk
        if mountpoint and mountpoint not in ['/', '/boot', '/home']:
            # Likely an external drive if mounted under /media or /mnt or removable
            if mountpoint.startswith('/media') or mountpoint.startswith('/mnt'):
                external_drives.append((mountpoint, name, fstype))

        # Check children (partitions)
        for child in device.get('children', []):
            self._extract_external_mounts(child, external_drives)

    def _detect_via_mount_points(self) -> List[Tuple[str, str, str]]:
        """Fallback method: Check common external drive mount points"""
        external_drives = []

        # Check /media directory
        media_path = Path('/media')
        if media_path.exists():
            for user_dir in media_path.iterdir():
                if user_dir.is_dir():
                    for drive_dir in user_dir.iterdir():
                        if drive_dir.is_dir() and os.access(drive_dir, os.W_OK):
                            external_drives.append((str(drive_dir), drive_dir.name, 'unknown'))

        # Check /mnt directory
        mnt_path = Path('/mnt')
        if mnt_path.exists():
            for drive_dir in mnt_path.iterdir():
                if drive_dir.is_dir() and os.access(drive_dir, os.W_OK) and drive_dir.name != 'wsl':
                    external_drives.append((str(drive_dir), drive_dir.name, 'unknown'))

        return external_drives

    def get_optimal_storage_path(self, subdir: str = "ech0_wisdom_data") -> Path:
        """
        Get the optimal storage path for ech0 training data
        Prefers external drive if available, falls back to local storage

        Args:
            subdir: Subdirectory name for ech0 data on the drive

        Returns:
            Path object for the optimal storage location
        """
        external_drives = self.detect_external_drives()

        if external_drives:
            # If preferred label specified, try to find it
            if self.preferred_label:
                for mount_point, device, fstype in external_drives:
                    if self.preferred_label.lower() in device.lower():
                        storage_path = Path(mount_point) / subdir
                        logger.info(f"✓ Found preferred external drive: {mount_point}")
                        return self._ensure_storage_path(storage_path, mount_point)

            # Otherwise use the first available external drive
            mount_point, device, fstype = external_drives[0]
            storage_path = Path(mount_point) / subdir
            logger.info(f"✓ Using external drive: {mount_point} ({device})")
            return self._ensure_storage_path(storage_path, mount_point)
        else:
            # No external drive found, use local storage
            logger.warning("⚠️  No external drive detected!")
            logger.info(f"📁 Using local storage: {self.default_local_path}")
            self.default_local_path.mkdir(exist_ok=True, parents=True)
            return self.default_local_path

    def _ensure_storage_path(self, storage_path: Path, mount_point: str) -> Path:
        """
        Ensure the storage path exists and is writable

        Args:
            storage_path: Path to create/verify
            mount_point: Mount point of the drive

        Returns:
            Verified storage path
        """
        try:
            storage_path.mkdir(exist_ok=True, parents=True)

            # Test write access
            test_file = storage_path / ".ech0_write_test"
            test_file.touch()
            test_file.unlink()

            logger.info(f"💾 Storage path ready: {storage_path}")
            return storage_path
        except (PermissionError, OSError) as e:
            logger.error(f"❌ Cannot write to external drive {mount_point}: {e}")
            logger.info(f"📁 Falling back to local storage: {self.default_local_path}")
            self.default_local_path.mkdir(exist_ok=True, parents=True)
            return self.default_local_path

    def monitor_and_report(self):
        """Monitor external drive status and report"""
        logger.info("\n" + "="*80)
        logger.info("🔍 External Drive Detection Report")
        logger.info("="*80)

        external_drives = self.detect_external_drives()

        if external_drives:
            logger.info(f"\n✓ Found {len(external_drives)} external drive(s):")
            for i, (mount_point, device, fstype) in enumerate(external_drives, 1):
                # Get available space
                try:
                    stat = os.statvfs(mount_point)
                    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
                    total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
                    used_percent = ((total_gb - free_gb) / total_gb * 100) if total_gb > 0 else 0

                    logger.info(f"\n  [{i}] {mount_point}")
                    logger.info(f"      Device: {device}")
                    logger.info(f"      Type: {fstype}")
                    logger.info(f"      Space: {free_gb:.1f} GB free / {total_gb:.1f} GB total ({used_percent:.1f}% used)")
                except Exception as e:
                    logger.info(f"\n  [{i}] {mount_point}")
                    logger.info(f"      Device: {device}")
                    logger.info(f"      Type: {fstype}")
                    logger.info(f"      (Could not retrieve space info: {e})")

            optimal_path = self.get_optimal_storage_path()
            logger.info(f"\n💾 Recommended storage path: {optimal_path}")
        else:
            logger.warning("\n⚠️  No external drives detected")
            logger.info("   Please connect an external drive to /media or /mnt")
            logger.info(f"   Currently using local storage: {self.default_local_path}")

        logger.info("\n" + "="*80 + "\n")


def get_wisdom_storage_path(preferred_label: Optional[str] = None) -> Path:
    """
    Quick helper function to get the wisdom storage path

    Args:
        preferred_label: Optional preferred drive label

    Returns:
        Path to use for storing wisdom/training data
    """
    manager = ExternalDriveManager(preferred_label=preferred_label)
    return manager.get_optimal_storage_path()


if __name__ == "__main__":
    # Run monitoring report
    manager = ExternalDriveManager()
    manager.monitor_and_report()

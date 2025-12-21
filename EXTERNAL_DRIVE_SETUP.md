# External Drive Streaming for ech0 Wisdom Ingestion

## Overview

The ech0 wisdom ingestion and training system now automatically detects and streams training data to external drives when connected. This feature ensures that large dataset generation (1M+ samples) doesn't fill up your local storage.

## Features

- **Automatic Detection**: Automatically detects external drives mounted at `/media` or `/mnt`
- **Smart Fallback**: Falls back to local storage (`./ech0_training_data`) if no external drive is found
- **Preferred Drive Support**: Can prioritize drives with specific labels (e.g., "ech0", "ech0_drive")
- **Space Monitoring**: Reports available space on detected drives
- **Seamless Integration**: Works with all ech0 training scripts without manual configuration

## How It Works

When you run any of the wisdom ingestion/training scripts, the system will:

1. Check for external drives mounted at `/media` or `/mnt`
2. If found, stream all training data to `<external_drive>/ech0_wisdom_data/`
3. If not found, use local directory `./ech0_training_data`

## Supported Scripts

The following scripts now support automatic external drive detection:

- `ech0_dataset_generator.py` - Core dataset generator
- `generate_1m_dataset.py` - 1M+ sample generation
- `ech0_train_orchestrator.py` - Complete training pipeline
- `ech0_finetune_engine.py` - Fine-tuning engine

## Usage

### Basic Usage (Automatic Detection)

Simply run your training scripts as normal:

```bash
# Generate 1M+ dataset - automatically uses external drive if connected
python3 generate_1m_dataset.py

# Run full training pipeline - automatically uses external drive if connected
python3 ech0_train_orchestrator.py
```

### Check External Drive Status

To see what external drives are detected:

```bash
python3 ech0_external_drive_manager.py
```

This will display:
- All detected external drives
- Mount points
- Available space
- Recommended storage path

### Advanced Usage

#### Disable External Drive Detection

If you want to force local storage:

```bash
python3 generate_1m_dataset.py --no-external-drive
```

#### Specify Custom Output Directory

Override automatic detection with a specific path:

```bash
python3 generate_1m_dataset.py --output-dir /path/to/custom/location
```

#### Use Preferred Drive Label

The system can prioritize drives with specific labels. By default, it looks for drives labeled "ech0" or containing "ech0" in the device name.

## Setting Up a Dedicated External Drive

For best results, we recommend setting up a dedicated external drive for ech0 wisdom data:

### Option 1: Label Your Drive "ech0"

**Linux/macOS:**
```bash
# For ext4 filesystem
sudo e2label /dev/sdX1 ech0

# For exFAT/FAT32
sudo fatlabel /dev/sdX1 ech0
```

**Windows:**
- Right-click drive in File Explorer
- Select "Rename"
- Enter "ech0"

### Option 2: Mount at Preferred Location

**Linux:**
```bash
# Create mount point
sudo mkdir -p /mnt/ech0_drive

# Mount drive
sudo mount /dev/sdX1 /mnt/ech0_drive

# Make permanent (add to /etc/fstab)
echo "/dev/sdX1 /mnt/ech0_drive ext4 defaults 0 2" | sudo tee -a /etc/fstab
```

**macOS:**
Drives automatically mount to `/Volumes/` - rename your drive to "ech0" for automatic detection.

## Storage Requirements

Recommended storage for ech0 wisdom ingestion:

- **1M sample dataset**: ~5-10 GB
- **10M sample dataset**: ~50-100 GB
- **Full training with checkpoints**: ~100-500 GB (depending on model size)

We recommend a drive with at least **500 GB** of free space for complete training runs.

## Troubleshooting

### Drive Not Detected

If your external drive isn't being detected:

1. **Check if it's mounted:**
   ```bash
   df -h | grep -E '/media|/mnt'
   ```

2. **Check permissions:**
   ```bash
   ls -la /media/$USER/
   ls -la /mnt/
   ```

3. **Try mounting manually:**
   ```bash
   sudo mount /dev/sdX1 /mnt/ech0_drive
   ```

### Permission Denied Errors

If you get permission errors:

```bash
# Make drive writable
sudo chmod 777 /path/to/external/drive

# Or change ownership
sudo chown -R $USER:$USER /path/to/external/drive
```

### Data Already on Local Storage

If you started training before connecting an external drive:

```bash
# Move existing data to external drive
mv ./ech0_training_data/* /path/to/external/drive/ech0_wisdom_data/

# Create symlink for compatibility
rm -rf ./ech0_training_data
ln -s /path/to/external/drive/ech0_wisdom_data ./ech0_training_data
```

## How to Check if It's Working

When you run a training script, you should see output like:

```
🔍 Checking for external drive...
✓ Found external drive: /media/user/ech0_drive (sdb1)
💾 Storage path ready: /media/user/ech0_drive/ech0_wisdom_data
```

If no external drive is found:

```
🔍 Checking for external drive...
⚠️  No external drive detected
📁 Using local storage: ./ech0_training_data
```

## API Reference

For programmatic use:

```python
from ech0_external_drive_manager import get_wisdom_storage_path, ExternalDriveManager

# Get optimal storage path
storage_path = get_wisdom_storage_path(preferred_label="ech0")

# Or use the manager directly
manager = ExternalDriveManager(preferred_label="ech0")
manager.monitor_and_report()
optimal_path = manager.get_optimal_storage_path()
```

## Support

If you encounter issues with external drive detection, please:

1. Check the troubleshooting section above
2. Run `python3 ech0_external_drive_manager.py` to diagnose
3. Report issues with full output from the diagnostic script

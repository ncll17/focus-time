# focus-time
Focus time experimentation

## Prerequisites

- Python 3.12
- PDM (Python Development Master) for dependency management

## Installing PDM

### Windows

1. Using pip:

```powershell
pip install pdm
```

2. Using winget:
```powershell
winget install pdm
```

3. Using Scoop:
```powershell
scoop install pdm
```

### macOS

1. Using Homebrew:
```bash
brew install pdm
```

2. Using pip:
```bash
pip install pdm
```

### Ubuntu/Debian

1. Using apt (recommended):
```bash
curl -sSL https://pdm.fming.dev/dev/install-pdm.py | python3 -
```

2. Using pip:
```bash
pip install pdm
```

### Verify Installation

After installation, verify PDM is correctly installed:
```bash
pdm --version
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/focus-time.git
cd focus-time
```

2. Initialize PDM project:
```bash
pdm init
```

3. Install dependencies:
```bash
pdm install
```

## Development Setup

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality and consistency. The hooks check for:
- Code formatting (Black)
- Import sorting (isort) - TODO
- Code style (flake8) - TODO
- Basic file hygiene (trailing whitespace, YAML validation, etc.) - TODO

#### Setup Pre-commit

1. Add pre-commit to your project:
```bash
pdm add pre-commit --dev
```

2. Install dependencies:
```bash
pdm install
```

3. Install the pre-commit hooks:
```bash
pdm run pre-commit install
```

4. (Optional) Test the hooks on all files:
```bash
pdm run pre-commit run --all-files
```

#### Troubleshooting Pre-commit Installation

If you encounter "Command 'pre-commit' is not found in your PATH", try these alternative installation methods:

##### Windows
```powershell
# Using pip
pip install pre-commit

# Using winget
winget install pre-commit
```

##### macOS
```bash
# Using Homebrew
brew install pre-commit
```

##### Ubuntu
```bash
# Using apt
sudo apt install pre-commit
```

After installing pre-commit globally, run:
```bash
pre-commit install
```

### Git Hooks in Action

The pre-commit hooks will automatically run on every commit. If any checks fail:
1. The commit will be blocked
2. The hooks will attempt to fix issues automatically where possible
3. You'll need to stage any automatic fixes (`git add .`)
4. Try committing again

## Training the Model

The training script (`train.py`) uses a transformer-based architecture to learn patterns in application usage sequences.

### Configuration

Before running the training script, ensure you have:
- Raw data file with application usage data
- Application mappings file
- Configuration file at `config/train/default.yaml`

### Running Training

To start training:
```bash
pdm run python train.py
```

### Monitoring

Training progress can be monitored through:
- Console logs using Loguru
- TensorBoard visualizations (logs saved in `runs/` directory)

To view TensorBoard metrics:
```bash
pdm run tensorboard --logdir runs
```

This README provides:
- Clear prerequisites
- Setup instructions using PDM
- Basic instructions for running the training script
- Information about monitoring training progress

Feel free to adjust the content based on any specific requirements or additional information you'd like to include!
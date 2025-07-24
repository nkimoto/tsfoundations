# TSFoundations

A Python project with uv, ruff, and cursor configuration

## Features

- Modern Python development with uv
- Code formatting and linting with ruff
- Type checking with mypy
- Testing with pytest
- Cursor AI integration

## Requirements

- Python 3.13 or higher
- uv (Python package manager)
- ruff (linter and formatter)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd tsfoundations
```

2. Install dependencies with uv:
```bash
uv sync
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
uv sync --dev

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

### Code Quality

```bash
# Format code with ruff
ruff format

# Lint code with ruff
ruff check

# Type checking with mypy
mypy tsfoundations

# Run all quality checks
ruff check --fix
```

### Testing

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=tsfoundations

# Run tests in watch mode
pytest-watch
```

### Building

```bash
# Build package
uv build

# Install in development mode
uv pip install -e .
```

## Usage

Add your main application code to the `tsfoundations/` directory.

Example:
```python
# tsfoundations/main.py
def main():
    print("Hello from TSFoundations!")

if __name__ == "__main__":
    main()
```

Then run:
```bash
python -m tsfoundations.main
```

## Project Structure

```
tsfoundations/
├── tsfoundations/          # Main package
│   └── __init__.py
├── tests/                    # Test files
│   └── __init__.py
├── pyproject.toml           # Project configuration
└── README.md               # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and quality checks
5. Submit a pull request

## License

This project is licensed under the MIT License. 
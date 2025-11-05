# Contributing to ThermoFleet-eVTOL-Simulator

Thank you for your interest in contributing to ThermoFleet-eVTOL-Simulator! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/lalomorales22/ThermoFleet-eVTOL-Simulator/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs. actual behavior
   - System information (OS, GPU, Python version)
   - Relevant logs or screenshots

### Suggesting Features

1. Check existing issues and discussions
2. Create a feature request issue with:
   - Clear description of the feature
   - Use cases and benefits
   - Possible implementation approach (if you have ideas)

### Pull Requests

1. **Fork the repository**

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the code style (PEP 8)
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   # Run tests
   pytest tests/

   # Check code style
   black src/ scripts/
   flake8 src/ scripts/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: Clear description of changes"
   ```

   Commit message format:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Update:` for improvements
   - `Docs:` for documentation
   - `Test:` for tests

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template with:
     - Description of changes
     - Related issues
     - Testing done
     - Screenshots (if UI changes)

## Development Setup

1. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/FlyingCarRL.git
   cd FlyingCarRL
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up pre-commit hooks** (optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Code Style

- Follow [PEP 8](https://pep8.org/) for Python code
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and modular

### Example:
```python
def spawn_evtol(
    vehicle_type: str,
    position: Tuple[float, float, float],
    count: int = 1
) -> List[Vehicle]:
    """
    Spawn eVTOL vehicles in the arena.

    Args:
        vehicle_type: Type of vehicle (small, medium, large)
        position: Initial position (x, y, z) in meters
        count: Number of vehicles to spawn

    Returns:
        List of spawned Vehicle objects

    Raises:
        ValueError: If vehicle_type is invalid
    """
    # Implementation...
```

## Testing

- Write tests for new functionality
- Place tests in `tests/` directory
- Use pytest for testing
- Aim for >80% code coverage

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_vehicles.py::test_spawn_small_evtol
```

## Documentation

- Update README.md if adding major features
- Add docstrings to new functions/classes
- Update relevant guides in `docs/`
- Add examples for new features

## Project Structure

```
FlyingCarRL/
├── src/               # Source code
│   ├── environments/  # Arena and environment code
│   ├── vehicles/      # Vehicle models and physics
│   ├── sensors/       # Sensor simulation
│   ├── training/      # RL training code
│   └── utils/         # Utilities
├── scripts/           # Setup and utility scripts
├── docs/              # Documentation
├── assets/            # 3D models, configs, scenes
├── data/              # Database, logs, episodes
└── tests/             # Test suite
```

## Areas Needing Contribution

Priority areas (see Development Roadmap in README.md):

- [ ] Phase 2: Vehicle modeling and spawning
- [ ] Phase 3: RL integration and training
- [ ] Phase 4: Visual frontend development
- [ ] Phase 5: Database logging and analytics
- [ ] Phase 6: Testing and optimization

Other areas:
- Adding new vehicle types
- Improving physics models
- Performance optimization
- Documentation and tutorials
- Example scenarios and benchmarks

## Questions?

- Open a [Discussion](https://github.com/yourusername/FlyingCarRL/discussions)
- Check existing [Issues](https://github.com/yourusername/FlyingCarRL/issues)
- Review the [README](README.md) and [documentation](docs/)

## Recognition

Contributors will be recognized in:
- README.md Acknowledgments section
- Release notes
- Contributors page (coming soon)

Thank you for contributing to FlyingCarRL! 🚁

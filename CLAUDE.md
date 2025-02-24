# CLAUDE.md - RAG Project Guidelines

## Build/Test Commands
- Run scripts: `python <filename>.py`
- Activate venv: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt` (if created)
- Single test: `pytest path/to/test.py -v`

## Code Style Guidelines
- **Imports**: Group standard library, third-party, and local imports in that order
- **Formatting**: Use consistent indentation (4 spaces), max line length ~80-100 chars
- **Types**: Use type hints in function signatures
- **Naming**: 
  - Snake_case for variables and functions
  - CamelCase for classes
  - UPPERCASE for constants  
- **Documentation**: Use docstrings for all functions, following Google style
- **Error Handling**: Use try/except blocks with specific exceptions
- **Code Organization**: Keep functions focused and modular, limit side effects

## Project Structure
- Core RAG techniques in separate modules
- Utilities in helper_utils.py
- Data in data/ directory
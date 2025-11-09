# Reusable Prompts for Common Development Tasks

This directory contains reusable prompt files for GitHub Copilot to assist with specific, recurring tasks in this project.

## Available Prompts

### Flask Route Development
**File:** `flask-route.prompt.md`  
**Use when:** Creating new API endpoints for the Flask web application  
**Topics:** REST API, request validation, error handling, JSON responses

### Service Module Development
**File:** `service-module.prompt.md`  
**Use when:** Creating new service modules for external API integration  
**Topics:** API clients, retry logic, rate limiting, error handling

### Type Hint Conversion
**File:** `type-hints.prompt.md`  
**Use when:** Converting old code to use precise type hints  
**Topics:** Python 3.11+ type hints, built-in generics, type annotations

### Refactoring
**File:** `refactor.prompt.md`  
**Use when:** Refactoring code to improve structure  
**Topics:** Code organization, best practices, maintainability

## How to Use

1. Open the relevant prompt file
2. Read the context and requirements
3. Use Copilot with the prompt context loaded
4. Follow the generated suggestions and checklist

## Project-Specific Guidelines

### This Project Uses:
- **Python 3.11+**: Built-in generics (`list[]`, `dict[]` not `List`, `Dict`)
- **Flask**: Web framework for API endpoints
- **Google Gemini AI**: For transcript section generation
- **Poetry**: Dependency management
- **pytest**: Testing framework

### Common Patterns:
- Service pattern for external APIs (Gemini, DeepL, etc.)
- Retry logic with exponential backoff
- Comprehensive logging with `logger`
- JSON-based API responses with `jsonify()`
- Google Style docstrings

## Creating New Prompts

When creating a new prompt file:
1. Use `.prompt.md` extension
2. Include clear task description
3. Provide code examples matching project patterns
4. List requirements and constraints
5. Include testing examples
6. Reference project coding guidelines in `copilot-instructions.md`


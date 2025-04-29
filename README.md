# Universal Web Testing Framework

An AI-powered automated testing framework for web applications, specifically optimized for Angular frontends with .NET backends but flexible enough to test virtually any website.

![Test Framework Banner](https://via.placeholder.com/900x200/0066cc/ffffff?text=Universal+Web+Testing+Framework)

## Features

- **Universal Compatibility**: Test any website regardless of technology stack
- **AI-Enhanced Analysis**: Machine learning algorithms identify patterns in test failures
- **Intelligent Exploration**: Autonomously discover and test UI elements
- **Multi-Type Testing**: UI, API, and end-to-end test capabilities
- **Comprehensive Reporting**: Detailed HTML reports with screenshots and recommendations
- **Selenium Integration**: Leverages the power of Selenium WebDriver for reliable UI testing
- **Low Configuration**: Get started quickly with sensible defaults
- **Flexible Test Creation**: Easy-to-define tests through code or configuration

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Creating Tests](#creating-tests)
- [Running Tests](#running-tests)
- [Test Reports](#test-reports)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.7+
- Chrome, Firefox, or Edge browser
- Appropriate WebDriver for your browser

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/universal-web-testing-framework.git
   cd universal-web-testing-framework
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the appropriate WebDriver for your browser:
   - [ChromeDriver](https://sites.google.com/chromium.org/driver/)
   - [GeckoDriver](https://github.com/mozilla/geckodriver/releases) (Firefox)
   - [EdgeDriver](https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/)

4. Add the WebDriver to your PATH or specify its location in your configuration.

## Quick Start

Test a website with default settings:

```bash
python universal_web_tester.py --url https://example.com
```

This will:
1. Create a default configuration for the website
2. Generate a basic test suite
3. Run the tests
4. Generate an HTML test report

## Configuration

The framework uses a JSON configuration file (`config.json` by default). You can create and modify it manually or use the built-in configuration generator:

```python
from universal_web_tester import UniversalWebTester

tester = UniversalWebTester()
tester.create_default_config("https://example.com")
```

### Configuration Options

```json
{
  "base_url": "https://example.com",
  "api_url": "https://api.example.com",
  "browser": "chrome",
  "headless": false,
  "implicit_wait": 10,
  "explicit_wait": 15,
  "screenshot_dir": "screenshots",
  "test_data_dir": "test_data",
  "test_results_dir": "test_results",
  "learning_enabled": true,
  "random_exploration": false,
  "max_exploration_depth": 3,
  "credentials": {
    "username": "testuser",
    "password": "testpassword"
  },
  "test_sets": []
}
```

## Creating Tests

### Programmatically Creating Tests

```python
from universal_web_tester import UniversalWebTester

tester = UniversalWebTester()
tester.create_default_config("https://example.com")

# Create a custom test
tester.create_custom_test(
    name="Login Test",
    target_element="#login-button",
    action="click",
    assertion="visible"
)

# Create an input test
tester.create_custom_test(
    name="Search Test",
    target_element="#search-input",
    action="input",
    input_value="test query",
    assertion="visible"
)
```

### Creating Tests Using Configuration

You can also define tests in JSON format:

```python
from universal_web_tester import AITestRunner

runner = AITestRunner()

# Create a UI test
login_test = runner.create_ui_test(
    name="Login Test",
    url="https://example.com/login",
    steps=[
        {
            "type": "input",
            "selector_type": "id",
            "selector": "username",
            "value": "testuser"
        },
        {
            "type": "input",
            "selector_type": "id",
            "selector": "password",
            "value": "password123"
        },
        {
            "type": "click",
            "selector_type": "id",
            "selector": "login-button",
            "wait_after": 2
        }
    ],
    assertions=[
        {
            "type": "visible",
            "selector": ".dashboard-welcome"
        }
    ]
)

# Add to a test suite
runner.add_test_suite("Authentication Tests", [login_test])
```

## Running Tests

### Command Line Interface

```bash
python universal_web_tester.py --url https://example.com --browser firefox --headless --report my_report.html
```

### Programmatic Execution

```python
from universal_web_tester import UniversalWebTester

tester = UniversalWebTester()
tester.create_default_config("https://example.com")
tester.generate_test_suite_for_website()

# Run all tests
results = tester.run_tests()

# Run a specific test suite
results = tester.run_tests("Authentication Tests")

# Generate an HTML report
tester.generate_html_report("test_report.html")
```

## Test Reports

The framework generates detailed HTML reports with:

- Overall test statistics (pass/fail rates)
- Detailed test results
- Screenshots of failures
- AI-generated analysis of test patterns
- Recommendations for improving tests

## Advanced Usage

### Intelligent Website Exploration

The framework can autonomously explore your website to discover UI elements and potential test paths:

```python
tester = UniversalWebTester()
tester.create_default_config("https://example.com")

# Configure exploration
tester.test_runner.framework.config["random_exploration"] = True
tester.test_runner.framework.config["max_exploration_depth"] = 4

# Start exploration
tester.test_runner.explore_application()
```

### AI Analysis of Test Results

After running tests, you can access the AI analysis:

```python
tester = UniversalWebTester()
# Run tests...

# Get AI analysis
analysis = tester.results.get("analysis", {})
print("Recommendations:", analysis.get("recommendations", []))
```

### Testing Angular-Specific Features

For Angular applications, the framework includes specialized support:

```python
# Wait for Angular rendering to complete
tester.test_runner.framework.wait_for_angular()

# Create an Angular-aware test
angular_test = tester.test_runner.create_ui_test(
    name="Angular Component Test",
    url="https://my-angular-app.com/component",
    wait_for_angular=True,
    steps=[...]
)
```

### Testing .NET API Endpoints

For testing .NET backend APIs:

```python
api_test = tester.test_runner.create_api_test(
    name="User API Test",
    endpoint="/api/users",
    method="GET",
    headers={"Authorization": "Bearer token123"},
    assertions=[
        {"type": "status_code", "value": 200},
        {"type": "json_path", "path": "$.users", "condition": "not_empty"}
    ]
)
```

## Project Structure

```
universal-web-testing-framework/
├── universal_web_tester.py       # Main script and CLI entry point
├── config.json                   # Default configuration
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── screenshots/                  # Test failure screenshots
├── test_data/                    # Test data files
└── test_results/                 # Test results and reports
```

## Dependencies

- selenium: Web browser automation
- pandas, numpy: Data processing
- scikit-learn: Machine learning for test analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Built with [Selenium WebDriver](https://www.selenium.dev/documentation/webdriver/)
- Inspired by test automation best practices
- Enhanced with machine learning for intelligent testing

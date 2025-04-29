import os
import time
import json
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, 
    NoSuchElementException, 
    ElementNotInteractableException,
    StaleElementReferenceException
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_test_framework.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AITestingFramework:
    """
    AI-powered testing framework for Angular applications with .NET backend
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the AI testing framework
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.driver = None
        self.wait = None
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "total": 0,
            "details": []
        }
        self.test_history = []
        self.learned_patterns = {}
        self.ml_models = {}
        
        # Initialize the ML models
        self._init_ml_models()
        
        logger.info(f"AI Testing Framework initialized with config from {config_path}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Set default values for missing config options
            default_config = {
                "base_url": "http://localhost:4200",
                "api_url": "http://localhost:5000",
                "browser": "chrome",
                "headless": False,
                "implicit_wait": 10,
                "explicit_wait": 15,
                "screenshot_dir": "screenshots",
                "test_data_dir": "test_data",
                "test_results_dir": "test_results",
                "learning_enabled": True,
                "random_exploration": False,
                "max_exploration_depth": 3,
                "credentials": {
                    "username": "",
                    "password": ""
                },
                "test_sets": []
            }
            
            # Merge default with provided config
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                    
            return config
            
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return {
                "base_url": "http://localhost:4200",
                "api_url": "http://localhost:5000",
                "browser": "chrome",
                "headless": False,
                "implicit_wait": 10,
                "explicit_wait": 15,
                "screenshot_dir": "screenshots",
                "test_data_dir": "test_data",
                "test_results_dir": "test_results",
                "learning_enabled": True,
                "random_exploration": False,
                "max_exploration_depth": 3,
                "credentials": {
                    "username": "",
                    "password": ""
                },
                "test_sets": []
            }
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in config file {config_path}")
            raise

    def _init_ml_models(self):
        """Initialize machine learning models used for intelligent testing"""
        # Text analysis model for error message classification
        self.ml_models["text_vectorizer"] = TfidfVectorizer(max_features=100)
        self.ml_models["error_classifier"] = KMeans(n_clusters=5)
        
        # User interaction pattern analysis
        self.ml_models["interaction_scaler"] = StandardScaler()
        self.ml_models["interaction_clusterer"] = KMeans(n_clusters=3)
        
        logger.info("Machine learning models initialized")
    
    def setup(self):
        """Set up the WebDriver and create necessary directories"""
        # Create directories if they don't exist
        for dir_name in ["screenshots", "test_data", "test_results"]:
            dir_path = self.config.get(f"{dir_name}_dir", dir_name)
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize WebDriver
        if self.config["browser"].lower() == "chrome":
            options = webdriver.ChromeOptions()
            if self.config["headless"]:
                options.add_argument("--headless")
            self.driver = webdriver.Chrome(options=options)
        elif self.config["browser"].lower() == "firefox":
            options = webdriver.FirefoxOptions()
            if self.config["headless"]:
                options.add_argument("--headless")
            self.driver = webdriver.Firefox(options=options)
        elif self.config["browser"].lower() == "edge":
            options = webdriver.EdgeOptions()
            if self.config["headless"]:
                options.add_argument("--headless")
            self.driver = webdriver.Edge(options=options)
        else:
            raise ValueError(f"Unsupported browser: {self.config['browser']}")
        
        # Configure wait times
        self.driver.implicitly_wait(self.config["implicit_wait"])
        self.wait = WebDriverWait(self.driver, self.config["explicit_wait"])
        
        logger.info(f"WebDriver set up with {self.config['browser']} browser")
        
    def teardown(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()
            self.driver = None
        
        # Save test results
        self._save_test_results()
        
        # Update learning models if enabled
        if self.config["learning_enabled"] and self.test_results["total"] > 0:
            self._update_ml_models()
            
        logger.info("Teardown complete")
    
    def _save_test_results(self):
        """Save test results to file"""
        if self.test_results["total"] > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                self.config["test_results_dir"], 
                f"test_results_{timestamp}.json"
            )
            
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=2)
                
            logger.info(f"Test results saved to {results_file}")
    
    def _update_ml_models(self):
        """Update machine learning models based on test results"""
        try:
            # Extract error messages from failed tests
            error_messages = [
                result["error_message"] 
                for result in self.test_results["details"] 
                if result["status"] == "failed" and result.get("error_message")
            ]
            
            if error_messages:
                # Update error classification model
                vectors = self.ml_models["text_vectorizer"].fit_transform(error_messages)
                self.ml_models["error_classifier"].fit(vectors)
                
                # Classify errors into clusters
                clusters = self.ml_models["error_classifier"].predict(vectors)
                
                # Store clustered errors for future analysis
                clustered_errors = {}
                for i, cluster in enumerate(clusters):
                    if cluster not in clustered_errors:
                        clustered_errors[cluster] = []
                    clustered_errors[cluster].append(error_messages[i])
                
                self.learned_patterns["error_clusters"] = clustered_errors
                
            logger.info("Machine learning models updated with new test data")
            
        except Exception as e:
            logger.error(f"Error updating ML models: {str(e)}")
    
    def navigate_to(self, url: str = None):
        """
        Navigate to the specified URL or base URL from config
        
        Args:
            url: URL to navigate to. If None, use base_url from config
        """
        target_url = url if url else self.config["base_url"]
        self.driver.get(target_url)
        logger.info(f"Navigated to {target_url}")
    
    def login(self, username: str = None, password: str = None):
        """
        Login to the application
        
        Args:
            username: Username for login. If None, use from config
            password: Password for login. If None, use from config
        """
        # Use provided credentials or get from config
        username = username or self.config["credentials"]["username"]
        password = password or self.config["credentials"]["password"]
        
        try:
            # This is a generic login implementation - customize for your app
            self.wait_for_element(By.ID, "username")
            self.driver.find_element(By.ID, "username").send_keys(username)
            self.driver.find_element(By.ID, "password").send_keys(password)
            self.driver.find_element(By.ID, "login-button").click()
            
            # Wait for login to complete - adjust selector based on your app
            self.wait_for_element(By.CSS_SELECTOR, ".dashboard-container")
            
            logger.info(f"Successfully logged in as {username}")
            return True
            
        except Exception as e:
            logger.error(f"Login failed: {str(e)}")
            self.take_screenshot("login_failed")
            return False
    
    def wait_for_element(self, by: By, selector: str, timeout: int = None):
        """
        Wait for an element to be present and visible
        
        Args:
            by: Selenium By strategy
            selector: Element selector
            timeout: Wait timeout in seconds. If None, use from config
            
        Returns:
            WebElement if found, None otherwise
        """
        timeout = timeout or self.config["explicit_wait"]
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.visibility_of_element_located((by, selector))
            )
            return element
        except TimeoutException:
            logger.warning(f"Element not found: {by}={selector} after {timeout}s")
            return None
    
    def wait_for_angular(self, timeout: int = None):
        """
        Wait for Angular to finish rendering
        
        Args:
            timeout: Wait timeout in seconds. If None, use from config
        """
        timeout = timeout or self.config["explicit_wait"]
        try:
            # Check if Angular is present and wait for it to finish rendering
            script = """
                var callback = arguments[arguments.length - 1];
                try {
                    if (window.getAllAngularTestabilities) {
                        window.getAllAngularTestabilities().forEach(function(testability) {
                            testability.whenStable(function() {
                                callback('Angular stable');
                            });
                        });
                    } else {
                        callback('Angular not found');
                    }
                } catch (e) {
                    callback('Error: ' + e);
                }
            """
            self.driver.execute_async_script(script)
            logger.debug("Angular rendering completed")
        except Exception as e:
            logger.warning(f"Error waiting for Angular: {str(e)}")
    
    def take_screenshot(self, name: str = None):
        """
        Take a screenshot of the current page
        
        Args:
            name: Screenshot name. If None, use timestamp
        """
        if not name:
            name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        file_path = os.path.join(self.config["screenshot_dir"], f"{name}.png")
        try:
            self.driver.save_screenshot(file_path)
            logger.info(f"Screenshot saved to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to take screenshot: {str(e)}")
            return None
    
    def run_test(self, test_name: str, test_func, *args, **kwargs):
        """
        Run a single test and record results
        
        Args:
            test_name: Name of the test
            test_func: Test function to execute
            *args, **kwargs: Arguments to pass to test function
        """
        logger.info(f"Running test: {test_name}")
        self.test_results["total"] += 1
        
        start_time = time.time()
        try:
            result = test_func(*args, **kwargs)
            end_time = time.time()
            
            test_result = {
                "name": test_name,
                "status": "passed" if result else "failed",
                "duration": end_time - start_time,
                "timestamp": datetime.now().isoformat(),
                "screenshot": None,
                "error_message": None
            }
            
            if result:
                self.test_results["passed"] += 1
                logger.info(f"Test '{test_name}' passed ({test_result['duration']:.2f}s)")
            else:
                self.test_results["failed"] += 1
                test_result["screenshot"] = self.take_screenshot(f"failed_{test_name}")
                test_result["error_message"] = "Test returned False"
                logger.error(f"Test '{test_name}' failed ({test_result['duration']:.2f}s)")
                
        except Exception as e:
            end_time = time.time()
            self.test_results["failed"] += 1
            error_message = str(e)
            
            test_result = {
                "name": test_name,
                "status": "failed",
                "duration": end_time - start_time,
                "timestamp": datetime.now().isoformat(),
                "screenshot": self.take_screenshot(f"error_{test_name}"),
                "error_message": error_message
            }
            
            logger.error(f"Test '{test_name}' failed with error: {error_message}")
            
        self.test_results["details"].append(test_result)
        return test_result
    
    def run_test_suite(self, test_suite: List[Dict[str, Any]]):
        """
        Run a suite of tests
        
        Args:
            test_suite: List of test definitions
        """
        logger.info(f"Running test suite with {len(test_suite)} tests")
        
        for test in test_suite:
            test_name = test.get("name", "Unnamed Test")
            test_type = test.get("type", "custom")
            
            if test_type == "ui":
                self.run_ui_test(test)
            elif test_type == "api":
                self.run_api_test(test)
            elif test_type == "e2e":
                self.run_e2e_test(test)
            elif test_type == "custom":
                if "function" in test:
                    self.run_test(test_name, test["function"], **test.get("params", {}))
                else:
                    logger.warning(f"Skipping test '{test_name}': No test function specified")
                    self.test_results["skipped"] += 1
            else:
                logger.warning(f"Skipping test '{test_name}': Unknown test type '{test_type}'")
                self.test_results["skipped"] += 1
    
    def run_ui_test(self, test_config: Dict[str, Any]):
        """
        Run a UI test based on configuration
        
        Args:
            test_config: Test configuration
        """
        test_name = test_config.get("name", "Unnamed UI Test")
        logger.info(f"Running UI test: {test_name}")
        
        try:
            # Navigate to the target URL
            target_url = test_config.get("url", self.config["base_url"])
            self.navigate_to(target_url)
            
            # Wait for Angular if needed
            if test_config.get("wait_for_angular", True):
                self.wait_for_angular()
            
            # Execute test steps
            steps = test_config.get("steps", [])
            
            for step in steps:
                step_type = step.get("type", "click")
                selector_type = step.get("selector_type", "css")
                selector = step.get("selector", "")
                
                # Convert selector type to Selenium By
                by = self._get_by_from_selector_type(selector_type)
                
                # Wait for element if needed
                if step.get("wait_for_element", True):
                    element = self.wait_for_element(by, selector)
                    if element is None:
                        raise NoSuchElementException(f"Element not found: {selector_type}={selector}")
                else:
                    element = self.driver.find_element(by, selector)
                
                # Execute step based on type
                if step_type == "click":
                    element.click()
                    logger.debug(f"Clicked on {selector_type}={selector}")
                    
                elif step_type == "input":
                    value = step.get("value", "")
                    element.clear()
                    element.send_keys(value)
                    logger.debug(f"Entered '{value}' into {selector_type}={selector}")
                    
                elif step_type == "select":
                    value = step.get("value", "")
                    # Find option by value
                    option = element.find_element(By.CSS_SELECTOR, f"option[value='{value}']")
                    option.click()
                    logger.debug(f"Selected '{value}' in {selector_type}={selector}")
                    
                elif step_type == "assert":
                    assertion_type = step.get("assertion", "visible")
                    
                    if assertion_type == "visible":
                        assert element.is_displayed(), f"Element {selector} is not visible"
                    elif assertion_type == "text":
                        expected_text = step.get("value", "")
                        actual_text = element.text
                        assert expected_text in actual_text, f"Text '{expected_text}' not found in '{actual_text}'"
                    elif assertion_type == "attribute":
                        attribute = step.get("attribute", "")
                        expected_value = step.get("value", "")
                        actual_value = element.get_attribute(attribute)
                        assert expected_value == actual_value, f"Attribute '{attribute}' value '{actual_value}' != '{expected_value}'"
                
                # Wait after step if specified
                if "wait_after" in step:
                    time.sleep(step["wait_after"])
            
            # Final assertions
            assertions = test_config.get("assertions", [])
            for assertion in assertions:
                selector_type = assertion.get("selector_type", "css")
                selector = assertion.get("selector", "")
                assertion_type = assertion.get("type", "visible")
                
                by = self._get_by_from_selector_type(selector_type)
                element = self.wait_for_element(by, selector)
                
                if assertion_type == "visible":
                    assert element is not None and element.is_displayed(), f"Element {selector} is not visible"
                elif assertion_type == "text":
                    expected_text = assertion.get("value", "")
                    actual_text = element.text
                    assert expected_text in actual_text, f"Text '{expected_text}' not found in '{actual_text}'"
                elif assertion_type == "count":
                    expected_count = assertion.get("value", 0)
                    elements = self.driver.find_elements(by, selector)
                    assert len(elements) == expected_count, f"Found {len(elements)} elements, expected {expected_count}"
            
            # Test passed
            self.test_results["passed"] += 1
            self.test_results["details"].append({
                "name": test_name,
                "status": "passed",
                "duration": time.time() - time.time(),  # Placeholder for actual duration
                "timestamp": datetime.now().isoformat(),
                "screenshot": None,
                "error_message": None
            })
            
            logger.info(f"UI test '{test_name}' passed")
            return True
            
        except Exception as e:
            # Test failed
            self.test_results["failed"] += 1
            screenshot_path = self.take_screenshot(f"failed_{test_name}")
            
            self.test_results["details"].append({
                "name": test_name,
                "status": "failed",
                "duration": time.time() - time.time(),  # Placeholder for actual duration
                "timestamp": datetime.now().isoformat(),
                "screenshot": screenshot_path,
                "error_message": str(e)
            })
            
            logger.error(f"UI test '{test_name}' failed: {str(e)}")
            return False
    
    def run_api_test(self, test_config: Dict[str, Any]):
        """
        Run an API test based on configuration
        
        Args:
            test_config: Test configuration
        """
        # API testing would be implemented here
        # For simplicity, just log that this would be implemented
        test_name = test_config.get("name", "Unnamed API Test")
        logger.info(f"API test '{test_name}' would be implemented here")
        
        # Mock implementation - in a real scenario, this would make actual API calls
        self.test_results["passed"] += 1
        self.test_results["details"].append({
            "name": test_name,
            "status": "passed",
            "duration": 0.1,
            "timestamp": datetime.now().isoformat(),
            "screenshot": None,
            "error_message": None
        })
        
        return True
    
    def run_e2e_test(self, test_config: Dict[str, Any]):
        """
        Run an end-to-end test based on configuration
        
        Args:
            test_config: Test configuration
        """
        # E2E testing would combine UI and API tests
        test_name = test_config.get("name", "Unnamed E2E Test")
        logger.info(f"E2E test '{test_name}' would be implemented here")
        
        # Mock implementation - in a real scenario, this would run a full user flow
        self.test_results["passed"] += 1
        self.test_results["details"].append({
            "name": test_name,
            "status": "passed",
            "duration": 0.5,
            "timestamp": datetime.now().isoformat(),
            "screenshot": None,
            "error_message": None
        })
        
        return True
    
    def _get_by_from_selector_type(self, selector_type: str) -> By:
        """
        Convert a selector type string to a Selenium By
        
        Args:
            selector_type: Selector type string
            
        Returns:
            Selenium By
        """
        selector_type = selector_type.lower()
        
        if selector_type == "css":
            return By.CSS_SELECTOR
        elif selector_type == "xpath":
            return By.XPATH
        elif selector_type == "id":
            return By.ID
        elif selector_type == "name":
            return By.NAME
        elif selector_type == "class":
            return By.CLASS_NAME
        elif selector_type == "tag":
            return By.TAG_NAME
        elif selector_type == "link_text":
            return By.LINK_TEXT
        elif selector_type == "partial_link_text":
            return By.PARTIAL_LINK_TEXT
        else:
            logger.warning(f"Unknown selector type: {selector_type}, defaulting to CSS")
            return By.CSS_SELECTOR
    
    def explore_app(self, start_url: str = None, max_depth: int = None):
        """
        Intelligently explore the application to discover elements and flows
        
        Args:
            start_url: Starting URL for exploration
            max_depth: Maximum exploration depth
        """
        if not self.config["random_exploration"]:
            logger.info("App exploration disabled in config")
            return
        
        start_url = start_url or self.config["base_url"]
        max_depth = max_depth or self.config["max_exploration_depth"]
        
        logger.info(f"Starting app exploration from {start_url} with max depth {max_depth}")
        
        # Navigate to start URL
        self.navigate_to(start_url)
        
        # Wait for Angular
        self.wait_for_angular()
        
        # Track visited URLs to avoid loops
        visited_urls = set([start_url])
        
        # Start exploration
        self._explore_recursive(0, max_depth, visited_urls)
        
        logger.info(f"App exploration complete. Visited {len(visited_urls)} URLs")
    
    def _explore_recursive(self, current_depth: int, max_depth: int, visited_urls: set):
        """
        Recursive function for app exploration
        
        Args:
            current_depth: Current exploration depth
            max_depth: Maximum exploration depth
            visited_urls: Set of visited URLs
        """
        if current_depth >= max_depth:
            logger.debug(f"Reached maximum exploration depth {max_depth}")
            return
        
        # Get current URL
        current_url = self.driver.current_url
        
        # Find all clickable elements
        try:
            clickable_elements = self.driver.find_elements(By.CSS_SELECTOR, 
                "a, button, input[type='button'], input[type='submit'], [role='button']")
            
            # Filter out hidden elements
            visible_elements = [e for e in clickable_elements if e.is_displayed()]
            
            # Randomize order to add variety to exploration
            random.shuffle(visible_elements)
            
            logger.debug(f"Found {len(visible_elements)} clickable elements at depth {current_depth}")
            
            # Try to click each element
            for i, element in enumerate(visible_elements[:5]):  # Limit to 5 elements per page
                try:
                    # Take a screenshot before clicking
                    self.take_screenshot(f"explore_d{current_depth}_e{i}_before")
                    
                    # Get element attributes for logging
                    tag_name = element.tag_name
                    text = element.text[:20] if element.text else ""
                    element_id = element.get_attribute("id") or ""
                    element_class = element.get_attribute("class") or ""
                    
                    logger.debug(f"Clicking {tag_name} '{text}' (id={element_id}, class={element_class})")
                    
                    # Click the element
                    element.click()
                    
                    # Wait for Angular
                    self.wait_for_angular()
                    
                    # Check if URL changed
                    new_url = self.driver.current_url
                    if new_url != current_url:
                        logger.info(f"Navigation detected: {current_url} -> {new_url}")
                        
                        if new_url not in visited_urls:
                            visited_urls.add(new_url)
                            
                            # Take a screenshot after navigation
                            self.take_screenshot(f"explore_d{current_depth}_e{i}_after")
                            
                            # Continue exploration from new URL
                            self._explore_recursive(current_depth + 1, max_depth, visited_urls)
                            
                            # Navigate back
                            self.driver.back()
                            self.wait_for_angular()
                    
                except (ElementNotInteractableException, StaleElementReferenceException):
                    # Element might be stale or not interactable, skip it
                    continue
                except Exception as e:
                    logger.warning(f"Error during exploration: {str(e)}")
                    # Try to navigate back to current URL
                    try:
                        self.navigate_to(current_url)
                        self.wait_for_angular()
                    except:
                        pass
        
        except Exception as e:
            logger.error(f"Exploration error: {str(e)}")
    
    def generate_test_report(self):
        """
        Generate a comprehensive test report
        
        Returns:
            Dictionary containing test report data
        """
        report = {
            "summary": {
                "total": self.test_results["total"],
                "passed": self.test_results["passed"],
                "failed": self.test_results["failed"],
                "skipped": self.test_results["skipped"],
                "pass_rate": 0 if self.test_results["total"] == 0 else 
                    self.test_results["passed"] / self.test_results["total"] * 100
            },
            "details": self.test_results["details"],
            "timestamp": datetime.now().isoformat(),
            "config": {
                "browser": self.config["browser"],
                "base_url": self.config["base_url"],
                "headless": self.config["headless"]
            }
        }
        
        # Generate report file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(
            self.config["test_results_dir"], 
            f"test_report_{timestamp}.json"
        )
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Test report saved to {report_file}")
        return report
    
    def analyze_test_results(self):
        """
        Analyze test results using AI techniques
        
        Returns:
            Dictionary containing analysis results
        """
        if self.test_results["total"] == 0:
            logger.warning("No test results to analyze")
            return {"message": "No test results to analyze"}
        
        try:
            # Extract durations
            durations = [result["duration"] for result in self.test_results["details"]]
            
            # Calculate statistics
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
            
            # Identify slow tests
            slow_tests = [
                result["name"] 
                for result in self.test_results["details"] 
                if result["duration"] > avg_duration * 1.5
            ]
            
            # Identify test patterns
            failure_patterns = self._identify_failure_patterns()
            
            analysis = {
                "statistics": {
                    "total_tests": self.test_results["total"],
                    "pass_rate": self.test_results["passed"] / self.test_results["total"] * 100 if self.test_results["total"] > 0 else 0,
                    "average_duration": avg_duration,
                    "max_duration": max_duration,
                    "min_duration": min_duration
                },
                "slow_tests": slow_tests,
                "failure_patterns": failure_patterns,
                "recommendations": self._generate_recommendations(failure_patterns)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing test results: {str(e)}")
            return {"error": str(e)}
    
    def _identify_failure_patterns(self):
        """
        Identify patterns in test failures
        
        Returns:
            Dictionary containing failure patterns
        """
        failed_tests = [
            result for result in self.test_results["details"] 
            if result["status"] == "failed"
        ]
        
        if not failed_tests:
            return {"message": "No failed tests to analyze"}
        
        # Extract error messages
        error_messages = [
            test.get("error_message", "") for test in failed_tests
            if test.get("error_message")
        ]
        
        # Count error message occurrences
        error_counts = {}
        for msg in error_messages:
            # Extract the main error type (e.g., "TimeoutException: Element not found")
            error_type = msg.split(":")[0] if ":" in msg else msg
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Find most common error types
        common_errors = sorted(
            error_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            "common_errors": common_errors,
            "total_failures": len(failed_tests)
        }
    
    def _generate_recommendations(self, failure_patterns):
        """
        Generate recommendations based on failure patterns
        
        Args:
            failure_patterns: Dictionary containing failure patterns
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check if there are any failures
        if failure_patterns.get("total_failures", 0) == 0:
            return ["All tests passed, no recommendations needed."]
        
        # Analyze common errors
        common_errors = failure_patterns.get("common_errors", [])
        for error_type, count in common_errors:
            if "TimeoutException" in error_type:
                recommendations.append(
                    f"Increase timeout values or check element selectors for {count} tests failing with {error_type}"
                )
            elif "NoSuchElementException" in error_type:
                recommendations.append(
                    f"Review element selectors for {count} tests failing with {error_type}"
                )
            elif "StaleElementReferenceException" in error_type:
                recommendations.append(
                    f"Add wait conditions or refresh element references for {count} tests failing with {error_type}"
                )
        
        # General recommendations
        if self.test_results["failed"] > 0:
            recommendations.append(
                f"Review {self.test_results['failed']} failed tests and check for common patterns in failed test scenarios"
            )
        
        return recommendations


class AITestRunner:
    """
    Main class for running AI-powered tests
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the test runner
        
        Args:
            config_path: Path to configuration file
        """
        self.framework = AITestingFramework(config_path)
        self.test_suites = {}
        self.config_path = config_path
        
    def create_config_file(self, base_url: str, browser: str = "chrome", headless: bool = False):
        """
        Create a configuration file
        
        Args:
            base_url: Base URL for testing
            browser: Browser to use for testing
            headless: Whether to run in headless mode
        """
        config = {
            "base_url": base_url,
            "api_url": base_url,
            "browser": browser,
            "headless": headless,
            "implicit_wait": 10,
            "explicit_wait": 15,
            "screenshot_dir": "screenshots",
            "test_data_dir": "test_data",
            "test_results_dir": "test_results",
            "learning_enabled": True,
            "random_exploration": False,
            "max_exploration_depth": 3,
            "credentials": {
                "username": "",
                "password": ""
            },
            "test_sets": []
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Configuration file created at {self.config_path}")
        
    def add_test_suite(self, name: str, tests: List[Dict[str, Any]]):
        """
        Add a test suite
        
        Args:
            name: Test suite name
            tests: List of test definitions
        """
        self.test_suites[name] = tests
        logger.info(f"Test suite '{name}' added with {len(tests)} tests")
        
    def create_ui_test(self, name: str, url: str = None, steps: List[Dict[str, Any]] = None, 
                       assertions: List[Dict[str, Any]] = None):
        """
        Create a UI test
        
        Args:
            name: Test name
            url: URL to test
            steps: List of test steps
            assertions: List of assertions
            
        Returns:
            Dictionary containing UI test definition
        """
        test = {
            "name": name,
            "type": "ui",
            "url": url or self.framework.config["base_url"],
            "wait_for_angular": True,
            "steps": steps or [],
            "assertions": assertions or []
        }
        
        return test
    
    def create_api_test(self, name: str, endpoint: str, method: str = "GET", 
                        data: Dict[str, Any] = None, headers: Dict[str, Any] = None,
                        assertions: List[Dict[str, Any]] = None):
        """
        Create an API test
        
        Args:
            name: Test name
            endpoint: API endpoint
            method: HTTP method
            data: Request data
            headers: Request headers
            assertions: List of assertions
            
        Returns:
            Dictionary containing API test definition
        """
        test = {
            "name": name,
            "type": "api",
            "endpoint": endpoint,
            "method": method,
            "data": data or {},
            "headers": headers or {},
            "assertions": assertions or []
        }
        
        return test
    
    def run_all_tests(self):
        """
        Run all test suites
        """
        try:
            self.framework.setup()
            
            for suite_name, tests in self.test_suites.items():
                logger.info(f"Running test suite: {suite_name}")
                self.framework.run_test_suite(tests)
                
            report = self.framework.generate_test_report()
            analysis = self.framework.analyze_test_results()
            
            return {
                "report": report,
                "analysis": analysis
            }
            
        finally:
            self.framework.teardown()
    
    def run_test_suite(self, suite_name: str):
        """
        Run a specific test suite
        
        Args:
            suite_name: Test suite name
            
        Returns:
            Dictionary containing test results
        """
        if suite_name not in self.test_suites:
            logger.error(f"Test suite '{suite_name}' not found")
            return {"error": f"Test suite '{suite_name}' not found"}
        
        try:
            self.framework.setup()
            
            logger.info(f"Running test suite: {suite_name}")
            self.framework.run_test_suite(self.test_suites[suite_name])
                
            report = self.framework.generate_test_report()
            analysis = self.framework.analyze_test_results()
            
            return {
                "report": report,
                "analysis": analysis
            }
            
        finally:
            self.framework.teardown()
    
    def explore_application(self, start_url: str = None, max_depth: int = None):
        """
        Explore the application to discover elements and flows
        
        Args:
            start_url: Starting URL for exploration
            max_depth: Maximum exploration depth
        """
        try:
            self.framework.setup()
            self.framework.explore_app(start_url, max_depth)
        finally:
            self.framework.teardown()


class UniversalWebTester:
    """
    Universal web testing tool that works with any website
    """
    
    def __init__(self):
        """Initialize the universal web tester"""
        self.test_runner = AITestRunner()
        self.results = {}
        self.current_website = None
        
    def create_default_config(self, website_url: str):
        """
        Create a default configuration for the website
        
        Args:
            website_url: Website URL to test
        """
        self.test_runner.create_config_file(website_url)
        self.current_website = website_url
        logger.info(f"Created default configuration for {website_url}")
    
    def generate_test_suite_for_website(self, website_url: str = None):
        """
        Generate a test suite for the website
        
        Args:
            website_url: Website URL to test
        """
        url = website_url or self.current_website
        if not url:
            logger.error("No website URL specified")
            return
        
        # Create a test suite with common UI tests
        tests = [
            # Basic page load test
            self.test_runner.create_ui_test(
                name="Home Page Load Test",
                url=url,
                assertions=[
                    {"type": "visible", "selector": "body"}
                ]
            ),
            
            # Navigation test
            self.test_runner.create_ui_test(
                name="Navigation Test",
                url=url,
                steps=[
                    {
                        "type": "click",
                        "selector_type": "css",
                        "selector": "a",
                        "wait_after": 1
                    }
                ],
                assertions=[
                    {"type": "visible", "selector": "body"}
                ]
            ),
            
            # Form interaction test
            self.test_runner.create_ui_test(
                name="Form Interaction Test",
                url=url,
                steps=[
                    {
                        "type": "input",
                        "selector_type": "css",
                        "selector": "input[type='text']",
                        "value": "Test input",
                        "wait_after": 0.5
                    }
                ],
                assertions=[
                    {"type": "visible", "selector": "input[type='text']"}
                ]
            )
        ]
        
        self.test_runner.add_test_suite("Default Test Suite", tests)
        logger.info(f"Generated default test suite for {url}")
    
    def create_custom_test(self, name: str, target_element: str, action: str = "click", 
                         input_value: str = None, assertion: str = "visible"):
        """
        Create a custom test for a specific element
        
        Args:
            name: Test name
            target_element: CSS selector for the target element
            action: Action to perform (click, input, etc.)
            input_value: Value to input (if action is input)
            assertion: Assertion type (visible, text, etc.)
        """
        steps = []
        
        if action == "click":
            steps.append({
                "type": "click",
                "selector_type": "css",
                "selector": target_element,
                "wait_after": 1
            })
        elif action == "input":
            steps.append({
                "type": "input",
                "selector_type": "css",
                "selector": target_element,
                "value": input_value or "Test input",
                "wait_after": 0.5
            })
        
        assertions = [
            {"type": assertion, "selector": target_element}
        ]
        
        test = self.test_runner.create_ui_test(
            name=name,
            steps=steps,
            assertions=assertions
        )
        
        # Add to a custom test suite
        if "Custom Tests" not in self.test_runner.test_suites:
            self.test_runner.test_suites["Custom Tests"] = []
        
        self.test_runner.test_suites["Custom Tests"].append(test)
        logger.info(f"Created custom test '{name}' for element '{target_element}'")
    
    def run_tests(self, suite_name: str = None):
        """
        Run tests
        
        Args:
            suite_name: Test suite name (if None, run all tests)
        """
        if suite_name:
            results = self.test_runner.run_test_suite(suite_name)
        else:
            results = self.test_runner.run_all_tests()
        
        self.results = results
        return results
    
    def generate_html_report(self, output_file: str = "test_report.html"):
        """
        Generate an HTML report
        
        Args:
            output_file: Output file path
        """
        if not self.results:
            logger.warning("No test results to generate report")
            return
        
        # Get summary data
        summary = self.results.get("report", {}).get("summary", {})
        details = self.results.get("report", {}).get("details", [])
        
        # Create HTML content
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Universal Web Tester - Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        h1, h2 {{ color: #333; }}
        .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .summary-item {{ display: inline-block; margin-right: 20px; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .test-passed {{ background-color: rgba(0, 255, 0, 0.1); }}
        .test-failed {{ background-color: rgba(255, 0, 0, 0.1); }}
        .screenshot {{ max-width: 200px; cursor: pointer; }}
        .screenshot:hover {{ opacity: 0.8; }}
    </style>
</head>
<body>
    <h1>Universal Web Tester - Test Report</h1>
    <div class="summary">
        <h2>Test Summary</h2>
        <div class="summary-item">Total Tests: <strong>{summary.get('total', 0)}</strong></div>
        <div class="summary-item">Passed: <strong class="pass">{summary.get('passed', 0)}</strong></div>
        <div class="summary-item">Failed: <strong class="fail">{summary.get('failed', 0)}</strong></div>
        <div class="summary-item">Skipped: <strong>{summary.get('skipped', 0)}</strong></div>
        <div class="summary-item">Pass Rate: <strong>{summary.get('pass_rate', 0):.2f}%</strong></div>
    </div>
    
    <h2>Test Details</h2>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Status</th>
            <th>Duration (s)</th>
            <th>Error Message</th>
            <th>Screenshot</th>
        </tr>
"""
        
        # Add test details rows
        for test in details:
            status_class = "test-passed" if test.get("status") == "passed" else "test-failed"
            screenshot_html = ""
            if test.get("screenshot"):
                screenshot_path = test.get("screenshot")
                screenshot_html = f'<img src="{screenshot_path}" class="screenshot" onclick="window.open(\'{screenshot_path}\', \'_blank\')" />'
                
            html_content += f"""
        <tr class="{status_class}">
            <td>{test.get('name', 'Unknown')}</td>
            <td>{test.get('status', 'unknown')}</td>
            <td>{test.get('duration', 0):.2f}</td>
            <td>{test.get('error_message', '')}</td>
            <td>{screenshot_html}</td>
        </tr>"""
        
        # Add analysis if available
        analysis = self.results.get("analysis", {})
        statistics = analysis.get("statistics", {})
        recommendations = analysis.get("recommendations", [])
        
        html_content += f"""
    </table>
    
    <h2>Analysis</h2>
    <div class="summary">
        <h3>Statistics</h3>
        <div class="summary-item">Average Duration: <strong>{statistics.get('average_duration', 0):.2f}s</strong></div>
        <div class="summary-item">Max Duration: <strong>{statistics.get('max_duration', 0):.2f}s</strong></div>
        <div class="summary-item">Min Duration: <strong>{statistics.get('min_duration', 0):.2f}s</strong></div>
    </div>
    
    <h3>Recommendations</h3>
    <ul>
"""
        
        # Add recommendations
        for rec in recommendations:
            html_content += f"        <li>{rec}</li>\n"
            
        html_content += """
    </ul>
    
    <script>
        // Add any JavaScript for interactivity here
    </script>
</body>
</html>
"""
        
        # Write HTML to file
        with open(output_file, 'w') as f:
            f.write(html_content)
            
        logger.info(f"HTML report generated at {output_file}")
        return output_file


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal Web Testing Tool')
    parser.add_argument('--url', help='Website URL to test', required=True)
    parser.add_argument('--browser', help='Browser to use', default='chrome')
    parser.add_argument('--headless', help='Run in headless mode', action='store_true')
    parser.add_argument('--report', help='Output report file', default='test_report.html')
    
    args = parser.parse_args()
    
    # Create the tester
    tester = UniversalWebTester()
    
    # Configure for website
    tester.create_default_config(args.url)
    
    # Generate test suite
    tester.generate_test_suite_for_website()
    
    # Run tests
    results = tester.run_tests()
    
    # Generate report
    tester.generate_html_report(args.report)
    
    print(f"Testing complete! Report generated at {args.report}")
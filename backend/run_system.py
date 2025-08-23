#!/usr/bin/env python3
"""
Alpha Factor Trading System Startup Script
==========================================

This script starts the complete quantitative trading system:
1. FastAPI backend server
2. Optional: React frontend (if you have it set up)
3. Health checks and system validation

Usage:
    python run_system.py           # Start backend only
    python run_system.py --full    # Start backend + frontend (if available)
    python run_system.py --test    # Run system tests
"""

import os
import sys
import subprocess
import time
import requests
import argparse
from pathlib import Path

def print_banner():
    """Print system startup banner"""
    print("=" * 80)
    print("🚀 ALPHA FACTOR TRADING SYSTEM")
    print("=" * 80)
    print("🧠 Quantitative ML Pipeline | FastAPI Backend | React Frontend")
    print("💼 S&P 500 Alpha Factor Prediction with XGBoost + Random Forest")
    print("-" * 80)

def check_dependencies():
    """Check if required files and dependencies exist"""
    print("🔍 Checking system dependencies...")
    
    required_files = [
        "main.py",
        "utils/data_generator.py",
        "utils/alpha_factors.py", 
        "models/ml_models.py",
        "backtesting/backtesting_engine.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("   Please ensure all pipeline files are in the correct directories.")
        return False
    
    print("✅ All required files found")
    
    # Check Python packages
    required_packages = [
        "fastapi", "uvicorn", "pandas", "numpy", "scikit-learn", 
        "xgboost", "pydantic", "python-multipart"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing Python packages: {missing_packages}")
        print("   Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required packages installed")
    return True

def start_backend():
    """Start the FastAPI backend server"""
    print("🔧 Starting FastAPI backend server...")
    
    try:
        # Start uvicorn server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "backend:app", 
            "--host", "0.0.0.0",
            "--port", "8000", 
            "--reload"
        ])
        
        print("⏳ Waiting for backend to start...")
        time.sleep(3)
        
        # Health check
        try:
            response = requests.get("http://localhost:8000/", timeout=5)
            if response.status_code == 200:
                print("✅ Backend server started successfully!")
                print("🔗 API Health Check: http://localhost:8000/")
                print("📊 Swagger UI: http://localhost:8000/docs")
                print("📖 API Documentation: http://localhost:8000/redoc")
                return process
            else:
                print(f"❌ Backend health check failed: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Backend connection failed: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def test_pipeline():
    """Test the ML pipeline functionality"""
    print("🧪 Testing ML pipeline...")
    
    try:
        # Import and test main pipeline
        from main import main as run_pipeline
        print("✅ Pipeline import successful")
        
        # You could add more specific tests here
        print("✅ Pipeline ready for execution")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("🔌 Testing API endpoints...")
    
    base_url = "http://localhost:8000"
    endpoints = [
        "/",
        "/status", 
        "/debug/state"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint} - OK")
            else:
                print(f"❌ {endpoint} - Status {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint} - Error: {e}")

def start_frontend():
    """Start React frontend if available"""
    print("🎨 Checking for React frontend...")
    
    frontend_paths = ["frontend", "client", "ui", "."]
    package_json_found = False
    
    for path in frontend_paths:
        if Path(f"{path}/package.json").exists():
            print(f"✅ Found package.json in {path}")
            package_json_found = True
            
            try:
                print("📦 Installing frontend dependencies...")
                subprocess.run(["npm", "install"], cwd=path, check=True)
                
                print("🚀 Starting React development server...")
                process = subprocess.Popen(
                    ["npm", "start"], 
                    cwd=path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                print("⏳ Frontend starting... (this may take a moment)")
                time.sleep(5)
                print("✅ React frontend should be available at http://localhost:3000")
                return process
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Frontend startup failed: {e}")
                return None
            except FileNotFoundError:
                print("❌ npm not found. Please install Node.js and npm")
                return None
            break
    
    if not package_json_found:
        print("ℹ️  No React frontend found (package.json not found)")
        print("   You can still use the API directly or via Swagger UI")
        return None

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description="Alpha Factor Trading System Startup")
    parser.add_argument("--full", action="store_true", help="Start both backend and frontend")
    parser.add_argument("--test", action="store_true", help="Run system tests only") 
    parser.add_argument("--no-checks", action="store_true", help="Skip dependency checks")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Dependency checks
    if not args.no_checks:
        if not check_dependencies():
            print("\n❌ Dependency check failed. Please fix the issues above.")
            return 1
    
    # Test mode
    if args.test:
        print("\n🧪 RUNNING SYSTEM TESTS")
        print("-" * 40)
        success = test_pipeline()
        if success:
            print("\n✅ All tests passed!")
            return 0
        else:
            print("\n❌ Tests failed!")
            return 1
    
    # Start backend
    print("\n🔧 STARTING BACKEND SERVER")
    print("-" * 40)
    backend_process = start_backend()
    
    if not backend_process:
        print("❌ Failed to start backend server")
        return 1
    
    # Test API endpoints
    time.sleep(2)
    test_api_endpoints()
    
    # Start frontend if requested
    frontend_process = None
    if args.full:
        print("\n🎨 STARTING FRONTEND")
        print("-" * 40)
        frontend_process = start_frontend()
    
    # System ready
    print("\n" + "=" * 80)
    print("🎯 SYSTEM READY!")
    print("=" * 80)
    print("🔗 Backend API: http://localhost:8000")
    print("📊 Swagger UI: http://localhost:8000/docs") 
    print("🔧 Debug Info: http://localhost:8000/debug/state")
    if frontend_process:
        print("🎨 Frontend UI: http://localhost:3000")
    print("\n💡 To start the ML pipeline:")
    print("   1. Visit http://localhost:8000/docs")
    print("   2. Use POST /pipeline/start endpoint")
    print("   3. Monitor with GET /status endpoint")
    print("\n⌨️  Press Ctrl+C to stop all services")
    print("-" * 80)
    
    try:
        # Keep processes running
        while True:
            time.sleep(1)
            
            # Check if backend is still running
            if backend_process.poll() is not None:
                print("❌ Backend process died")
                break
                
            # Check if frontend is still running (if started)
            if frontend_process and frontend_process.poll() is not None:
                print("❌ Frontend process died")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down system...")
        
        if backend_process:
            backend_process.terminate()
            print("✅ Backend stopped")
            
        if frontend_process:
            frontend_process.terminate() 
            print("✅ Frontend stopped")
            
        print("👋 System shutdown complete")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
#!/usr/bin/env python3
"""
Debug version of run_system.py with better error reporting
"""

import os
import sys
import subprocess
import time
import threading
import queue
from pathlib import Path

def print_banner():
    print("🚀 Alpha Factor System - Debug Mode")
    print("-" * 50)

def check_api_file():
    """Check if api.py exists and has proper FastAPI setup"""
    if not Path("api.py").exists():
        print("❌ api.py not found")
        return False
    
    print("✅ Found api.py")
    
    # Check contents
    try:
        with open("api.py", "r") as f:
            content = f.read()
        
        if "FastAPI" in content:
            print("✅ FastAPI import found")
        else:
            print("⚠️ No FastAPI import found in api.py")
        
        if "app = FastAPI(" in content:
            print("✅ FastAPI app object found")
        else:
            print("⚠️ No 'app = FastAPI()' pattern found")
            
        return True
        
    except Exception as e:
        print(f"❌ Error reading api.py: {e}")
        return False

def test_imports():
    """Test if we can import the api module"""
    print("\n🔍 Testing imports...")
    
    try:
        # Try importing api module
        import api
        print("✅ Successfully imported api module")
        
        # Check if it has app attribute
        if hasattr(api, 'app'):
            print("✅ Found 'app' attribute in api module")
            return True
        else:
            print("❌ No 'app' attribute found in api module")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import api module: {e}")
        print(f"   Full error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing api import: {e}")
        return False

def test_dependencies():
    """Test if required dependencies are available"""
    print("\n🔍 Testing dependencies...")
    
    required_packages = ["fastapi", "uvicorn", "pydantic", "pandas", "numpy"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        print("   Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def capture_output(process, output_queue):
    """Capture process output in real-time"""
    for line in iter(process.stdout.readline, ''):
        if line:
            output_queue.put(('stdout', line.strip()))
    
    for line in iter(process.stderr.readline, ''):
        if line:
            output_queue.put(('stderr', line.strip()))

def start_api_with_detailed_logging():
    """Start the API with detailed error reporting"""
    print("\n🚀 Starting FastAPI server...")
    
    # Try multiple startup methods
    startup_commands = [
        # Method 1: Direct uvicorn module call
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
        # Method 2: Without reload
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"],
        # Method 3: Direct Python execution (if api.py has if __name__ == "__main__")
        [sys.executable, "api.py"],
    ]
    
    for i, cmd in enumerate(startup_commands, 1):
        print(f"\n⏳ Trying method {i}: {' '.join(cmd)}")
        
        try:
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            # Set up output capture
            output_queue = queue.Queue()
            stdout_thread = threading.Thread(target=capture_output, args=(process, output_queue))
            stdout_thread.daemon = True
            stdout_thread.start()
            
            # Monitor for startup success or failure
            startup_timeout = 10  # seconds
            start_time = time.time()
            
            while time.time() - start_time < startup_timeout:
                # Check if process died
                if process.poll() is not None:
                    # Process ended, collect all output
                    remaining_stdout, remaining_stderr = process.communicate()
                    if remaining_stdout:
                        print(f"📝 Final stdout: {remaining_stdout}")
                    if remaining_stderr:
                        print(f"❌ Final stderr: {remaining_stderr}")
                    print(f"❌ Process exited with code: {process.returncode}")
                    break
                
                # Check for output
                try:
                    while True:
                        output_type, line = output_queue.get_nowait()
                        print(f"📝 {output_type}: {line}")
                        
                        # Check for success indicators
                        if "Uvicorn running on" in line or "Application startup complete" in line:
                            print("✅ Server appears to be starting successfully!")
                            time.sleep(2)  # Give it a moment to fully start
                            return test_server_response(process)
                        
                        # Check for common error patterns
                        if any(error in line.lower() for error in ["error", "failed", "exception", "traceback"]):
                            print(f"❌ Error detected: {line}")
                            
                except queue.Empty:
                    pass
                
                time.sleep(0.1)
            
            # If we get here, either timeout or process ended
            if process.poll() is None:
                print("⏳ Server might be starting slowly, testing connection...")
                result = test_server_response(process)
                if result:
                    return result
                else:
                    print("❌ Server not responding, trying next method...")
                    process.terminate()
            
        except FileNotFoundError:
            print(f"❌ Command not found: {cmd[0]}")
        except Exception as e:
            print(f"❌ Error starting process: {e}")
    
    print("❌ All startup methods failed!")
    return None

def test_server_response(process=None):
    """Test if the server is responding"""
    print("\n🔍 Testing server response...")
    
    test_urls = [
        "http://localhost:8000/",
        "http://127.0.0.1:8000/", 
        "http://localhost:8000/docs",
        "http://localhost:8000/health"
    ]
    
    for url in test_urls:
        try:
            print(f"   Testing {url}...")
            
            # Try with requests if available
            try:
                import requests
                response = requests.get(url, timeout=5)
                print(f"   ✅ {url} -> Status: {response.status_code}")
                if response.status_code == 200:
                    print(f"   📄 Response: {response.text[:100]}...")
                    return process
            except ImportError:
                # Fallback to urllib
                import urllib.request
                import urllib.error
                
                with urllib.request.urlopen(url, timeout=5) as response:
                    content = response.read().decode()
                    print(f"   ✅ {url} -> Status: {response.getcode()}")
                    print(f"   📄 Response: {content[:100]}...")
                    return process
                    
        except Exception as e:
            print(f"   ❌ {url} -> {str(e)[:50]}...")
            continue
    
    print("❌ No endpoints responding")
    return None

def check_port_availability():
    """Check if port 8000 is available"""
    print("\n🔍 Checking port availability...")
    
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 8000))
        print("✅ Port 8000 is available")
        return True
    except OSError:
        print("❌ Port 8000 is already in use")
        
        # Try to find what's using the port
        try:
            result = subprocess.run(["lsof", "-i", ":8000"], capture_output=True, text=True)
            if result.stdout:
                print("📝 Process using port 8000:")
                print(result.stdout)
        except:
            try:
                result = subprocess.run(["netstat", "-an", "|", "grep", "8000"], capture_output=True, text=True, shell=True)
                if result.stdout:
                    print("📝 Port 8000 status:")
                    print(result.stdout)
            except:
                pass
        
        return False

def main():
    """Main debugging function"""
    print_banner()
    
    # Step 1: Check if api.py exists and is valid
    print("Step 1: Checking api.py...")
    if not check_api_file():
        print("❌ api.py check failed")
        return 1
    
    # Step 2: Test dependencies
    print("\nStep 2: Checking dependencies...")
    if not test_dependencies():
        print("❌ Dependencies check failed")
        return 1
    
    # Step 3: Test imports
    print("\nStep 3: Testing imports...")
    if not test_imports():
        print("❌ Import test failed")
        return 1
    
    # Step 4: Check port availability
    print("\nStep 4: Checking port availability...")
    port_available = check_port_availability()
    if not port_available:
        print("⚠️ Port 8000 is in use. The server might already be running!")
        print("   Try visiting http://localhost:8000 in your browser")
        
        # Test if it's our API
        existing_process = test_server_response()
        if existing_process:
            print("✅ API is already running and responding!")
            return 0
        else:
            print("❌ Something else is using port 8000")
            return 1
    
    # Step 5: Start the API server
    print("\nStep 5: Starting API server...")
    process = start_api_with_detailed_logging()
    
    if process:
        print("\n" + "="*60)
        print("🎉 SUCCESS! API is running!")
        print("="*60)
        print("🔗 Main API: http://localhost:8000")
        print("📚 Docs: http://localhost:8000/docs")
        print("🔧 Debug: http://localhost:8000/debug/state")
        print("="*60)
        print("\n💡 You can now:")
        print("   1. Visit http://localhost:8000/docs to see API documentation")
        print("   2. Test the health endpoint: curl http://localhost:8000/")
        print("   3. Start the ML pipeline: curl -X POST http://localhost:8000/pipeline/start")
        print("\n⌨️ Press Ctrl+C to stop the server...")
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
                if process.poll() is not None:
                    print("❌ Server stopped unexpectedly")
                    break
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping server...")
            process.terminate()
            print("✅ Server stopped")
        
        return 0
    else:
        print("\n❌ Failed to start API server")
        print("\n🔧 Manual troubleshooting steps:")
        print("   1. Run: python -c 'import api; print(\"API import OK\")'")
        print("   2. Run: python -m uvicorn api:app --reload")
        print("   3. Check: pip list | grep -E '(fastapi|uvicorn|pydantic)'")
        print("   4. Try: python api.py")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
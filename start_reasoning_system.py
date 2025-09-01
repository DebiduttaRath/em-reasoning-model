
#!/usr/bin/env python3
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

"""
Startup script for the Beast Reasoning LLM system.
This script starts both the backend API server and frontend UI.
"""

import subprocess
import time
import sys
import os
import signal
from pathlib import Path

def start_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting backend API server...")
    return subprocess.Popen([
        sys.executable, "server.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def start_frontend():
    """Start the Streamlit frontend."""
    print("🎨 Starting frontend UI...")
    return subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "ui.py",
        "--server.port", "5000",
        "--server.address", "0.0.0.0"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def wait_for_backend(max_attempts=30):
    """Wait for backend to be ready."""
    import requests
    
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://0.0.0.0:8000/health", timeout=1)
            if response.status_code == 200:
                print("✅ Backend is ready!")
                return True
        except:
            pass
        
        print(f"⏳ Waiting for backend... (attempt {attempt + 1}/{max_attempts})")
        time.sleep(2)
    
    return False

def main():
    """Main startup routine."""
    print("🧠 Beast Reasoning LLM System Startup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("server.py").exists() or not Path("ui.py").exists():
        print("❌ Error: server.py or ui.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend
        backend_process = start_backend()
        
        # Wait for backend to be ready
        if not wait_for_backend():
            print("❌ Backend failed to start properly!")
            sys.exit(1)
        
        # Start frontend
        frontend_process = start_frontend()
        time.sleep(3)  # Give frontend time to start
        
        print("\n🎉 System is ready!")
        print("=" * 50)
        print("📱 Frontend UI: http://localhost:5000")
        print("🔧 Backend API: http://localhost:8000")
        print("📖 API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop the system")
        print("=" * 50)
        
        # Keep the script running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend process died!")
                break
            
            if frontend_process.poll() is not None:
                print("❌ Frontend process died!")
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system...")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Clean up processes
        if backend_process and backend_process.poll() is None:
            print("🔄 Stopping backend...")
            backend_process.terminate()
            backend_process.wait()
        
        if frontend_process and frontend_process.poll() is None:
            print("🔄 Stopping frontend...")
            frontend_process.terminate()
            frontend_process.wait()
        
        print("✅ System shutdown complete!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import http.server
import socketserver
import webbrowser
import os
import threading
import time

def start_server():
    PORT = 8000
    
    # Change to the directory containing the HTML file
    os.chdir('/Users/pravinmenghani/Downloads/demos/llm/encode-decoder')
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🌐 Server starting at http://localhost:{PORT}")
        print(f"📄 Serving transformer_demo.html")
        print("🚀 Opening browser automatically...")
        print("Press Ctrl+C to stop the server")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(1)
            webbrowser.open(f'http://localhost:{PORT}/transformer_demo.html')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped!")
            httpd.shutdown()

if __name__ == "__main__":
    start_server()

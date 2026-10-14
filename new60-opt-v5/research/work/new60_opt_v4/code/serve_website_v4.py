"""Serve only the standalone V4 website on localhost; no external service."""
import argparse
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

ROOT=Path(__file__).resolve().parents[1]

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,directory=str(ROOT/'website'),**kwargs)
    def translate_path(self,path):
        if path.startswith('/new60-opt-v4/'):
            path='/'+path[len('/new60-opt-v4/'):]
        return super().translate_path(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8771)
    a=p.parse_args()
    if not (ROOT/'website/manifest.json').is_file():
        raise FileNotFoundError('A complete frozen V4 website is required; templates alone are not scientific results')
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    print(f'V4: http://127.0.0.1:{a.port}/new60-opt-v4/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()

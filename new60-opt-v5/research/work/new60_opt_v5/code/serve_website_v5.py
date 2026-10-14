"""Serve only the standalone V5 website on localhost; no external service."""
import argparse
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

ROOT=Path(__file__).resolve().parents[1]

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,directory=str(ROOT/'website'),**kwargs)
    def translate_path(self,path):
        if path.startswith('/new60-opt-v5/'):
            path='/'+path[len('/new60-opt-v5/'):]
        return super().translate_path(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8771)
    a=p.parse_args()
    if not (ROOT/'website/manifest.json').is_file():
        raise FileNotFoundError('A complete frozen V5 website is required; templates alone are not scientific results')
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    print(f'V5: http://127.0.0.1:{a.port}/new60-opt-v5/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()

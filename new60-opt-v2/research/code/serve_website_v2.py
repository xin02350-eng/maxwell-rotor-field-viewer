"""Serve both packaged versions locally without copying any scientific data."""
import argparse
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
ROOT=Path(__file__).resolve().parent.parent
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT/'website'),**kwargs)
 def translate_path(self,path):
  old=self.directory
  try:
   if path.startswith('/new60/'):
    self.directory=str(ROOT.parent/'new60_20261014/website');path='/'+path[len('/new60/'):]
   elif path.startswith('/new60-opt-v2/'):path='/'+path[len('/new60-opt-v2/'):]
   return super().translate_path(path)
  finally:self.directory=old
def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8769);a=p.parse_args()
 if not (ROOT/'website/manifest.json').exists():raise FileNotFoundError('Complete packaged website required')
 server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
 print(f'V2: http://127.0.0.1:{a.port}/\nV1: http://127.0.0.1:{a.port}/new60/',flush=True)
 try:server.serve_forever()
 except KeyboardInterrupt:pass
 finally:server.server_close()
if __name__=='__main__':main()

import logging
from gevent.pywsgi import WSGIServer

from app import create_app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app_server = WSGIServer(('0.0.0.0', 7000), app)
    app_server.serve_forever()

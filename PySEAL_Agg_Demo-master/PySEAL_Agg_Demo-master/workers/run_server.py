import argparse
import logging
import threading

from gevent.pywsgi import WSGIServer
from gevent import monkey

from app import create_app
from config.flask_config import WORKER_IP, DockerConfig, DefaultConfig

parser = argparse.ArgumentParser(description='Run the aggregator service')
parser.add_argument('--server_ip', dest='server_ip', action='store',
                    help='specify server ip to be used', default=None)
parser.add_argument('--aggregator_ip', dest='aggregator_ip', action='store',
                    help='specify aggregator ip to be used', default=None)
parser.add_argument('--is_docker', default=False, action='store_true',
                    help='specify whether docker IPs should be used')
results = parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if results.is_docker:
        config = DockerConfig
    else:
        config = DefaultConfig
    app1 = create_app(worker_id=1, config_object=config)
    app2 = create_app(worker_id=2, config_object=config)
    app_server1 = WSGIServer((WORKER_IP, 7101), app1)
    app_server2 = WSGIServer((WORKER_IP, 7102), app2)
    monkey.patch_thread()
    t = threading.Thread(target=app_server1.serve_forever)
    t.setDaemon(True)
    t.start()
    app_server2.serve_forever()

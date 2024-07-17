import threading


class ConnectionManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConnectionManager, cls).__new__(cls)
                    cls._instance._connections = {}
        return cls._instance

    def add_connection(self, provider_name, connection):
        self._connections[provider_name] = connection

    def get_connection(self, provider_name):
        return self._connections.get(provider_name)

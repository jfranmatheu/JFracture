import socket
import struct
from typing import Tuple, Union


class SocketSignal:
    STARTED = 0
    CONTINUE = 123
    PROGRESS = 100
    FINISHED = 1
    ERROR = 666
    REQUEST_SEND = 99
    WAIT = 111


class JFractureServer:
    address_family: socket.AddressFamily = socket.AF_INET
    socket_type   : socket.SocketKind    = socket.SOCK_STREAM

    host: str = 'localhost' # 127.0.0.1
    port: int = 0 # Automatically gets a free port.
    server_address: Tuple[str, int] = (host, port)

    def __init__(self, connections_count: int) -> None:
        self.connections_count = connections_count
        self.socket = socket.socket(self.address_family, self.socket_type)

    def __enter__(self) -> 'JFractureServer':
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def rcv_signal(self, connection: socket.SocketType) -> Union[Tuple[int, int], None]:
        # Get data from connection.
        data = connection.recv(struct.calcsize('i i'))

        # Small delay if no data was found.
        if data is None:
            return None

        try:
            # Get signal.
            unpacked_data = struct.unpack('i i', data)
            return unpacked_data
        except:
            return None
        
    def send_signal(self, connection: socket.SocketType, signal: SocketSignal) -> None:
        packed_data = struct.pack('i', signal)
        connection.send(packed_data)
        
    def new_connection(self):
        return self.socket.accept()

    def start(self):
        self.socket.bind(self.server_address)

        # Write ServerPort to config so that addon can connect safely.
        self.port = self.socket.getsockname()[1]
        print('[Server] Starting up on {} port {}'.format(self.host, self.port))
        self.socket.listen(self.connections_count)

    def stop(self):
        if self.socket:
            self.socket.close()
            self.socket = None

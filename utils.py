from dataclasses import dataclass, field 
import numpy as np 
import pickle
from typing import Union
import struct 
import socket
from PIL import Image
from enum import Enum

def receive_n_bytes(sock: socket.socket, n: int) -> Union[None, bytearray]:
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


@dataclass
class TextMessage: 
    text: str = None 
    requires_response: bool = True

@dataclass
class Message:
    image: Union[np.ndarray, Image.Image] = None
    text_message: Union[str, TextMessage] = '' 
    keyboard_command: int = None 
    requires_response: bool = True
    def __post_init__(self): 
        if isinstance(self.text_message, str): 
            self.text_message = TextMessage(text=self.text_message)
        if isinstance(self.image, Image.Image):
            self.image = np.asarray(self.image)

    @property
    def bytes(self): 
        self.image = Image.fromarray(self.image) if self.image is not None else self.image
        serialized = pickle.dumps(self)
        self.image = np.asarray(self.image)
        return serialized
    
    @classmethod 
    def reconstruct(cls, byte_repr: bytes): 
        msg: Message = pickle.loads(byte_repr)
        msg.image = np.asarray(msg.image)
        return msg


@dataclass
class Socket:
    socket: socket.socket
    def sendall(self, msg: bytes) -> None: 
        self.socket.sendall(msg)
    
    def send_message(self, msg: Union[str, TextMessage, Message]):
        if isinstance(msg, str) or isinstance(msg, TextMessage): 
            msg = Message(text=msg) 
        byte_message: bytes = msg.bytes 
        msg_length: int = len(byte_message)
        msg_length = struct.pack('>I', msg_length)

        self.socket.sendall(msg_length)
        self.socket.sendall(byte_message)

    def receive_message(self) -> Message: 
        msg_length = receive_n_bytes(self.socket, 4)
        msg_length = struct.unpack('>I', msg_length)[0]
        bytes_message = receive_n_bytes(self.socket, msg_length)
        return Message.reconstruct(bytes_message)


class Command(Enum): 
    ACCEPT = ord('s')
    FLAG = ord('f')
    MOVE_LEFT = 2 
    MOVE_RIGHT = 3 
    MOVE_LEFT_10 = ord('j')
    MOVE_RIGHT_10 = ord('l')
    MOVE_LEFT_100 = ord('u')
    MOVE_RIGHT_100 = ord('o')
    QUIT = ord('q')
    PASS = 0
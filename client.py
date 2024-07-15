import socket 
from config import SERVER_IP, PORT
from utils import TextMessage, Message, Socket, Command
from PIL import Image
import cv2
import numpy as np


def main(): 
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((SERVER_IP, PORT))
        client_socket = Socket(client)
        curr_image = np.zeros((480, 640 * 2, 3))
        while True: 
            msg = client_socket.receive_message()
            # print(msg)
            text_response = ''
            keyboard_command = Command.PASS

            if msg.text_message.text: 
                if msg.text_message.requires_response: 
                    text_response = input(msg.text_message.text)
                else: 
                    print(msg.text_message.text)
            
                response = Message(
                    text_message=text_response
                )
                client_socket.send_message(response)
                continue 

            if msg.image is not None: 
                curr_image = msg.image
            
            cv2.imshow('', curr_image[..., ::-1])
            command = cv2.waitKey(1) & 0xFF 
            try:
                command = Command(command)
            except ValueError: 
                command = Command.PASS
            response = Message(keyboard_command=command)
            client_socket.send_message(response)
            if command == Command.QUIT: 
                break 

if __name__ == '__main__': 
    main()
   
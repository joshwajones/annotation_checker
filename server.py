from config import SERVER_IP, PORT
import socket
import threading 
import logging
import os
import datetime
from dataclasses import dataclass, field 
from typing import Union 
from utils import Message, TextMessage, Socket, Command
import traceback
from dataset import DatasetDescription, DirectoryIterator, sort_trajectories
import cv2
from PIL import Image
import pickle
import glob
import numpy as np

SHOW_UNLOGGED_ONLY = True 
SAVE_EVERY = 5
INCLUDE_EVERY = 1
ALWAYS_START_UNLOGGED = True

DEFAULT_DATASET_DESCRIPTION = DatasetDescription( 
    dataset_directory='/home/paulzhou/octo_dataset_utils/check_annotations/datasets', 
    depth=4, 
    include_every=INCLUDE_EVERY
)
class Quit(Exception): 
    pass 

@dataclass
class Session: 
    client_socket: Socket
    save_ctdown: int = SAVE_EVERY

    def __post_init__(self): 
        self.logger = logging.getLogger('server')
        self.directory = dir_path = os.path.dirname(os.path.realpath(__file__)) 
        self.save_dir = os.path.join(dir_path, 'logs')
        self.user_name = self.get_user_name()
        
        
        self.log_directory = os.path.join(self.directory, 'logs', self.user_name)
        self.log_file = os.path.join(self.log_directory, 'log.pkl')
        os.makedirs(self.log_directory, exist_ok=True)
        if not os.path.exists(self.log_file): 
            with open(self.log_file, 'wb') as file: 
                pickle.dump({}, file)
        
        with open(self.log_file, 'rb') as file: 
            self.log = pickle.load(file)          

        self.trajectories = DirectoryIterator(DEFAULT_DATASET_DESCRIPTION).get_trajectory_list()
        if SHOW_UNLOGGED_ONLY: 
            traj_set = set(self.trajectories) - set(self.log.keys())
            self.trajectories = list(sort_trajectories(traj_set)) 
        self.set_start_point()
    
    def get_user_name(self) -> str: 
        available_users = list(sorted(os.listdir(self.save_dir)))
        text_lines = [] 
        
        text_lines = ['Available users:     ']
        text_lines.extend(f'{name}' for name in available_users)
        text = '\n'.join(text_lines)
        self.client_print(text)
        user_name = self.client_ask_confirm("Enter name from above list, or a new name:              ")
        return user_name 
    
    def set_start_point(self) -> None:
        num_completed = len(self.log)
        num_total  = len(self.trajectories)
        self.client_print(f'Checked {num_completed}/{num_total} trajectories!')
        if ALWAYS_START_UNLOGGED: 
            self.traj_index = 0 
        else: 
            response = self.client_ask_confirm('Continue from first un-checked trajectory?   (y/n)          ')
            if response == 'y': 
                for i, traj in enumerate(self.trajectories): 
                    if traj not in self.log: 
                        break 
                else: 
                    self.client_print("All trajectories checked! Quitting...")
                    raise Quit
                self.traj_index = i
            else: 
                self.traj_index = 0 

    def print_instruction(self) -> None: 
        info = [] 
        header_length = 50
        header_str = ' INSTRUCTIONS '
        header = '#' * ((header_length - len(header_str)) // 2) + header_str + '#' * ((header_length - len(header_str)) // 2)
        if len(header) < header_length: 
            header = header + "#"
        info.append(header)
        info.append(
            'Use the "j" to move left, and "l" to move right between trajectories, one at a time.'
        )
        info.append(
            'Use "u" and "o" to move between trajectories, 10 at a time.'
        )
        info.append('Or use "8" and "0" to move 100 at a time.\n')
        info.append(
            'For every trajectory, compare the logged target image to the actual image. If the target object is correct, press "s". '
            + '\nOtherwise, or if there is any doubt, or the target image is unclear, press "f".'
        )
        info.append('Then continue to the next trajectory.\n')
        info.append('Press "q" at any time to quit.')
        info.append('#' * header_length + '\n')
        self.client_print('\n'.join(info))

    def save(self) -> None: 
        with open(self.log_file, 'wb') as file: 
            pickle.dump(self.log, file)

    def get_and_combine_images(self, trajectory_path: str) -> Image.Image: 
        wrist_img_dir = os.path.join(trajectory_path, 'images1')
        def get_img_num_from_traj(traj): 
            im_str = traj.split('/')[-1]
            return int(im_str[len('im_'):im_str.find('.jpg')])
        wrist_images = list(sorted(
            glob.glob(os.path.join(wrist_img_dir, 'im_*.jpg')),
            key=get_img_num_from_traj
        ))
        curr_actual_path = wrist_images[-1]
        curr_actual = np.asarray(Image.open(curr_actual_path)) 

        curr_target_path = os.path.join(trajectory_path, 'target_image.png')
        if not os.path.exists(curr_target_path): 
            curr_target = np.zeros((480, 640, 3))
        else: 
            curr_target = np.asarray(Image.open(curr_target_path))
        h, w = curr_actual.shape[:2]
        if curr_actual.shape != curr_target.shape: 
            self.logger.warn(f'Incompatible actual/target shapes of actual={curr_actual.shape}, target={curr_target.shape} found. Resizing target...')
            
            curr_target = cv2.resize(curr_target, (w, h))

        combined_frame = np.zeros((h, 2 * w, 3))
        combined_frame[:, :w, :] = curr_actual
        combined_frame[:, w:, :] = curr_target
        combined_frame = np.uint8(combined_frame)
        combined_image = Image.fromarray(combined_frame)
        return combined_image 

    def execute_command(self, command: Command) -> None: 
        if command == Command.ACCEPT: 
            self.log[self.trajectories[self.traj_index]] = 'accept'
            self.save_ctdown -= 1 
        elif command == Command.FLAG: 
            self.log[self.trajectories[self.traj_index]] = 'flag'
            self.save_ctdown -= 1 
        elif command == Command.MOVE_LEFT: 
            self.traj_index -= 1 
            self.traj_index = max(self.traj_index, 0)
        elif command == Command.MOVE_RIGHT: 
            self.traj_index += 1 
            self.traj_index = min(len(self.trajectories) - 1, self.traj_index)
        elif command == Command.MOVE_LEFT_10: 
            self.traj_index -= 1 
            self.traj_index = max(self.traj_index, 0)
        elif command == Command.MOVE_RIGHT_10: 
            self.traj_index += 10
            self.traj_index = min(len(self.trajectories) - 1, self.traj_index)
        elif command == Command.MOVE_LEFT_100: 
            self.traj_index -= 100
            self.traj_index = max(self.traj_index, 0)
        elif command == Command.MOVE_RIGHT_100: 
            self.traj_index += 100
            self.traj_index = min(len(self.trajectories) - 1, self.traj_index)
        elif command == Command.QUIT: 
            raise Quit

    def check(self): 
        while True: 
            if self.save_ctdown == 0: 
                self.save()
                self.save_ctdown = SAVE_EVERY

            curr_traj = self.trajectories[self.traj_index]
            combined_image = self.get_and_combine_images(curr_traj)
            message = Message(
                image=combined_image
            )
            self.client_socket.send_message(message)

            response = self.client_socket.receive_message().keyboard_command

            self.execute_command(response)
            
    def client_print(self, text: str) -> None: 
        text_message = TextMessage(text=text, requires_response=False)
        msg = Message(text_message=text_message)
        self.client_socket.send_message(msg)
        self.client_socket.receive_message()
    
    def client_ask(self, text: str) -> str: 
        msg = Message(text_message=text)
        self.client_socket.send_message(msg)
        response = self.client_socket.receive_message().text_message.text 
        return response 

    def client_ask_confirm(self, text: str) -> str: 
        got_response = False 
        while not got_response: 
            response = self.client_ask(text)
            confirm_response = self.client_ask(f'Got response "{response}". Enter the same response to continue, or anything else to try again:      ')
            if confirm_response == response:
                got_response = True 

        return response 


def run_check(client_socket: Socket): 
    try:
        session = Session(client_socket=client_socket)
        session.print_instruction()
        session.check()
    except (Quit, TypeError, ConnectionResetError) as e: 
        session.save()
        

def accept_connections(): 
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:  
        server.bind((SERVER_IP, PORT))
        server.listen() 
        server.settimeout(None)
        while True: 
            client, address = server.accept()
            client_socket = Socket(client)
            threading.Thread(target=run_check, args=(client_socket,)).start()

        
def main(): 
    curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(dir_path, 'output', 'server', f'{curr_time}.log')
    logger = logging.getLogger('server')
    logging.basicConfig(
        filename=log_path, level=logging.INFO
    )
    try: 
        accept_connections()
    except (Exception, KeyboardInterrupt) as e: 
        logger.critical(f'RECEIVED EXCEPTION: {e}')
        logger.critical(f'{traceback.format_exc()}\n\n')


if __name__ == '__main__': 
    main() 


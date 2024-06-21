import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Message:
    def __init__(self, src: str, dst: str, data: bytes, tensor: torch.Tensor = None):
        self.src = src
        self.dst = dst
        self.data = data
        self.tensor = tensor
    def __str__(self):
        return f"{self.src} -> {self.dst}: {self.data}"
    def __repr__(self):
        return str(self)
    def __eq__(self, other):
        return self.src == other.src and self.dst == other.dst and self.data == other.data
    def __hash__(self):
        return hash((self.src, self.dst, self.data))
    def to_input_tensor(self) -> torch.Tensor:
        src_tensor = F.one_hot(torch.tensor([ord(self.src)]), num_classes=257).float()
        data_tensor = F.one_hot(torch.tensor([b for b in self.data]), num_classes=257).float()
        null_tensor = torch.cat([torch.zeros((256)), torch.ones((1))]).float()
        return torch.cat((src_tensor, null_tensor, data_tensor, null_tensor), dim=0)
    def to_output_tensor(self):
        dst_tensor = F.one_hot(torch.tensor([ord(self.dst)]), num_classes=257).float()
        data_tensor = F.one_hot(torch.tensor([b for b in self.data]), num_classes=257).float()
        null_tensor = torch.cat([torch.zeros((256)), torch.ones((1))]).float()
        return torch.cat((dst_tensor, null_tensor, data_tensor, null_tensor), dim=0)
    @staticmethod
    def from_output_tensor(tensor: torch.Tensor, src: str):
        dst_tensor = torch.tensor([])
        data_tensor = torch.tensor([])
        for c in tensor:
            if c[256] == 1:
                break
            dst_tensor = torch.cat((dst_tensor, c[:256]))
        for c in tensor:
            if c[256] == 1:
                break
            data_tensor = torch.cat((data_tensor, c[:256]))
        return Message(chr(torch.argmax(dst_tensor).item()), src, bytes([torch.argmax(data_tensor).item()]), tensor=tensor)
    @staticmethod
    def get_null_message(src: str):
        return Message(src, src, b"")
    @staticmethod
    def get_null_tensor():
        return torch.cat([torch.zeros((256)), torch.ones((1))]).float()
    @staticmethod
    def get_random_message(src: str, dst: str, length: int = random.randint(1, 256)):
        return Message(src, chr(random.randint(0, 255)), bytes([random.randint(0, 255) for _ in range(length)]))
    
from typing import Any

class Router:
    def __init__(self):
        self.interfaces: dict[str, Any] = {}
    def add_interface(self, name: str, interface):
        if name in self.interfaces:
            raise ValueError(f"Interface {name} already exists")
        if not hasattr(interface, "receive"):
            raise ValueError(f"Interface {name} does not have a receive method")
        self.interfaces[name] = interface
    def send(self, message: Message):
        self.interfaces[message.dst].receive(message)

class Agent:
    def __init__(self, name: str, model: nn.Module, router: Router):
        self.name = name
        self.model = model
        self.incoming: list[Message] = []
        self.router = router
        self.current_input_message = None
        self.current_input_index = 0
        self.current_output_message = None
        self.current_output_index = 0
        self.current_output_nulls = 0
        self.running = False
        self.paused = False
    def receive(self, message: Message):
        self.incoming.append(message)
    def get_next_c(self):
        if self.current_input_message is None:
            if len(self.incoming) == 0:
                return Message.get_null_tensor()
            self.current_input_message = self.incoming.pop(0).to_input_tensor()
        c = self.current_input_message[self.current_input_index]
        self.current_input_index += 1
        if self.current_input_index == len(self.current_input_message):
            self.current_input_message = None
            self.current_input_index = 0
        return c
    def add_new_c(self, c: torch.Tensor):
        if c.argmax() == 256:
            if self.current_output_nulls == 1:
                message = Message.from_output_tensor(self.current_output_message, self.name)
                self.router.send(message)
                self.current_output_message = None
                self.current_output_index = 0
                self.current_output_nulls = 0
            elif self.current_output_index != 0:
                self.current_output_nulls = 1
        else:
            if self.current_output_message is None:
                self.current_output_message = torch.stack([c])
            else:
                self.current_output_message = torch.cat([self.current_output_message, torch.stack([c])])
            self.current_output_index += 1
    def step(self):
        c = self.get_next_c()
        output = self.model(c)
        self.add_new_c(output)
    async def run(self):
        while self.running:
            if not self.paused:
                self.step()
    def start(self):
        self.running = True
        self.run()
    def stop(self):
        self.running = False
    def pause(self):
        self.paused = True
    def resume(self):
        self.paused = False
    
class EchoTrainer:
    def __init__(self, router: Router, name: str, agent_name: str):
        self.router = router
        self.name = name
        self.agent_name = agent_name
        self.running = False
        self.paused = False
    async def run(self):
        while self.running:
            if not self.paused:
                message = Message.get_random_message(self.agent_name)
                self.router.send(message)
    def start(self):
        self.running = True
        self.run()
    def stop(self):
        self.running = False
    def pause(self):
        self.paused = True
    def resume(self):
        self.paused = False
    def receive(self, message: Message):
        if message.dst == self.agent_name:
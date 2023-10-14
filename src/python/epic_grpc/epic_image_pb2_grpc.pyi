"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import abc
import collections.abc
import epic_image_pb2
import grpc

class epic_post_processStub:
    def __init__(self, channel: grpc.Channel) -> None: ...
    filter_and_save: grpc.UnaryUnaryMultiCallable[
        epic_image_pb2.epic_image,
        epic_image_pb2.empty,
    ]
    filter_and_save_chunk: grpc.StreamUnaryMultiCallable[
        epic_image_pb2.epic_image,
        epic_image_pb2.empty,
    ]

class epic_post_processServicer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def filter_and_save(
        self,
        request: epic_image_pb2.epic_image,
        context: grpc.ServicerContext,
    ) -> epic_image_pb2.empty: ...
    @abc.abstractmethod
    def filter_and_save_chunk(
        self,
        request_iterator: collections.abc.Iterator[epic_image_pb2.epic_image],
        context: grpc.ServicerContext,
    ) -> epic_image_pb2.empty: ...

def add_epic_post_processServicer_to_server(servicer: epic_post_processServicer, server: grpc.Server) -> None: ...
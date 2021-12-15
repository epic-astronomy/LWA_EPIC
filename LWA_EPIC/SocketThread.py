# -*- coding: utf-8 -*-

from __future__ import print_function

import threading
import socket
try:
    import queue
except NameError:
    import Queue as queue
    
class UDPRecvThread(threading.Thread):
    #STOP = '__UDPRecvThread_STOP__'
    def __init__(self, address, bufsize=16384):
        threading.Thread.__init__(self)
        self._addr      = address
        self._bufsize   = bufsize
        self._msg_queue = queue.Queue() # For default behaviour
        self.socket     = socket.socket(socket.AF_INET,
                                        socket.SOCK_DGRAM)
        self.socket.bind(address)
        self.stop_requested = threading.Event()
    def request_stop(self):
        """
        sendsock = socket.socket(socket.AF_INET,
                                socket.SOCK_DGRAM)
        sendsock.connect(self._addr)
        sendsock.send(UDPRecvThread.STOP)
        """
        self.stop_requested.set()
        # WAR for "107: Transpose endpoint is not connected" in socket.shutdown
        self.socket.connect(("0.0.0.0", 0))
        self.socket.shutdown(socket.SHUT_RD)
        self.socket.close()
    def run(self):
        while True:#not self.stop_requested.is_set():
            #pkt = self.socket.recv(self._bufsize)
            pkt, src_addr = self.socket.recvfrom(self._bufsize)
            if self.stop_requested.is_set():
                break
            #if pkt == UDPRecvThread.STOP:
            #	break
            src_ip = src_addr[0]
            self.process(pkt, src_ip)
        self.shutdown()
    def process(self, pkt, src_ip):
        """Overide this in subclass"""
        self._msg_queue.put((pkt,src_ip)) # Default behaviour
    def shutdown(self):
        """Overide this in subclass"""
        pass
    def get(self, timeout=None):
        try:
            return self._msg_queue.get(True, timeout)
        except queue.Empty:
            return None

if __name__ == '__main__':
    port = 8321
    rcv = UDPRecvThread(("localhost", port))
    #rcv.daemon = True
    rcv.start()
    print("Waiting for packet on port", port)
    pkt,src_ip = rcv.get(timeout=5.)
    if pkt is not None:
        print("Received packet:", pkt,src_ip)
    else:
        print("Timed out waiting for packet")
    rcv.request_stop()
    rcv.join()

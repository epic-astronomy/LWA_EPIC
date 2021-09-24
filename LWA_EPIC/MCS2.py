# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
try:
    range = xrange
    def data_to_hex(data):
        return data.encode('hex')
except NameError:
    def data_to_hex(data):
        try:
            return data.hex()
        except TypeError:
            return data.encode().hex()

try:
    import queue
except ImportError:
    import Queue as queue
import time
from datetime import datetime
from .ConsumerThread import ConsumerThread
from .SocketThread import UDPRecvThread
import string
import struct

import socket
from threading import Thread, Event, Semaphore

# Maximum number of bytes to receive from MCS
MCS_RCV_BYTES = 16*1024

# Note: Unless otherwise noted, slots are referenced to Unix time
def get_current_slot():
    # Returns current slot in Unix time
    return int(time.time())
def get_current_mpm():
    # Returns current milliseconds past midnight as an integer
    dt = datetime.utcnow()
    ms = int(dt.microsecond / 1000.)
    return ((dt.hour*60 + dt.minute)*60 + dt.second)*1000 + ms
def slot2utc(slot=None):
    if slot is None:
        slot = get_current_slot()
    return time.gmtime(slot)
# TODO: What is 'station time'? Is it UTC?
def slot2dayslot(slot=None):
    utc     = slot2utc(slot)
    dayslot = (utc.tm_hour*60 + utc.tm_min)*60 + utc.tm_sec
    return dayslot
def slot2mpm(slot=None):
    return slot2dayslot(slot) * 1000
def slot2mjd(slot=None):
    tt = slot2utc(slot)
    # Source: SkyField
    janfeb = tt.tm_mon < 3
    jd_day = tt.tm_mday
    jd_day += 1461 *  (tt.tm_year + 4800 - janfeb) // 4
    jd_day +=  367 *  (tt.tm_mon  -    2 + janfeb * 12) // 12
    jd_day -=    3 * ((tt.tm_year + 4900 - janfeb) // 100) // 4
    jd_day -= 32075
    mjd = tt.tm_sec
    mjd = mjd*(1./60) + tt.tm_min
    mjd = mjd*(1./60) + tt.tm_hour
    mjd = mjd*(1./24) + (jd_day - 2400000.5)
    mjd -= 0.5
    return mjd

def mib_parse_label(data):
    """Splits an MIB label into a list of arguments
        E.g., "ANT71_TEMP_MAX" --> ['ANT', 71, 'TEMP', 'MAX']
    """
    args = []
    arg = ''
    mode = None
    for c in data:
        if c in string.ascii_uppercase:
            if mode == 'i':
                args.append(int(arg))
                arg = ''
            arg += c
            mode = 's'
        elif c in string.digits:
            if mode == 's':
                args.append(arg)
                arg = ''
            arg += c
            mode = 'i'
        elif c == '_':
            if mode is not None:
                args.append(int(arg) if mode == 'i' else arg)
                arg = ''
            mode = None
    args.append(int(arg) if mode == 'i' else arg)
    key = mib_args2key(args)
    return key, args
def mib_args2key(args):
    """Merges an MIB label arg list back into a label suitable for
        use as a lookup key (all indexes are removed)."""
    return '_'.join([arg for arg in args if not isinstance(arg, int)])

class Msg(object):
    count = 0
    # Note: MsgSender will automatically set src
    def __init__(self, illegal_argument=None,
                src=None, dst=None, cmd=None, ref=None, data='', dst_ip=None,
                pkt=None, src_ip=None):
        assert(illegal_argument is None) # Ensure named args only
        self.dst  = dst
        self.src  = src
        self.cmd  = cmd
        self.ref  = ref
        if self.ref is None:
            self.ref = Msg.count % 10**9
            Msg.count += 1
        self.mjd  = None
        self.mpm  = None
        self.data = data
        self.dst_ip = dst_ip
        self.slot = None # For convenience, not part of encoded pkt
        if pkt is not None:
            self.decode(pkt)
        self.src_ip = src_ip
    def __str__(self):
        if self.slot is None:
            return ("<MCS Msg %i: '%s' from %s to %s, data=%r (0x%s)>" %
                    (self.ref, self.cmd, self.src, self.dst,
                     self.data, data_to_hex(self.data)))
        else:
            return (("<MCS Msg %i: '%s' from %s (%s) to %s, data=%r (0x%s), "+
                     "rcv'd in slot %i>") %
                     (self.ref, self.cmd, self.src, self.src_ip,
                      self.dst, self.data, data_to_hex(self.data),
                      self.slot))
    def decode(self, pkt):
        hdr = pkt[:38]
        try:
            hdr = hdr.decode()
        except Exception as e:
            # Python2 catch/binary data catch
            print('hdr error:', str(e), '@', hdr)
            pass

        self.slot = get_current_slot()
        self.dst  = hdr[:3]
        self.src  = hdr[3:6]
        self.cmd  = hdr[6:9]
        self.ref  = int(hdr[9:18])
        datalen   = int(hdr[18:22])
        self.mjd  = int(hdr[22:28])
        self.mpm  = int(hdr[28:37])
        space     = hdr[37]
        self.data = pkt[38:38+datalen]
        # WAR for DATALEN parameter being wrong for BAM commands (FST too?)
        broken_commands = ['BAM']#, 'FST']
        if self.cmd in broken_commands:
            self.data = pkt[38:]

    def create_reply(self, accept, status, data=''):
        msg = Msg(#src=self.dst,
                  dst=self.src,
                  cmd=self.cmd,
                  ref=self.ref,
                  dst_ip=self.src_ip)
        #msg.mjd, msg.mpm = getTime()
        response = 'A' if accept else 'R'
        msg.data = response + str(status).rjust(7)
        try:
            msg.data = msg.data.encode()
        except AttributeError:
            # Python2 catch
            pass
        try:
            data = data.encode()
        except AttributeError:
            # Python2 catch
            pass
        msg.data = msg.data+data
        return msg
    def is_valid(self):
        return (self.dst is not None and len(self.dst) <= 3 and
                self.src is not None and len(self.src) <= 3 and
                self.cmd is not None and len(self.cmd) <= 3 and
                self.ref is not None and (0 <= self.ref < 10**9) and
                self.mjd is not None and (0 <= self.mjd < 10**6) and
                self.mpm is not None and (0 <= self.mpm < 10**9) and
                len(self.data) < 10**4)
    def encode(self):
        self.mjd = int(slot2mjd())
        self.mpm = get_current_mpm()
        assert( self.is_valid() )
        pkt = (self.dst.ljust(3) +
               self.src.ljust(3) +
               self.cmd.ljust(3) +
               str(self.ref      ).rjust(9) +
               str(len(self.data)).rjust(4) +
               str(self.mjd      ).rjust(6) +
               str(self.mpm      ).rjust(9) +
               ' ')
        try:
            pkt = pkt.encode()
            self.data = self.data.encode()
        except (AttributeError, UnicdoeDecodeError):
            # Python2 catch
            pass
        return pkt+self.data

class MsgReceiver(UDPRecvThread):
    def __init__(self, address, subsystem='ALL'):
        UDPRecvThread.__init__(self, address)
        self.subsystem = subsystem
        self.msg_queue = queue.Queue()
        self.name      = 'MCS.MsgReceiver'
    def process(self, pkt, src_ip):
        if len(pkt):
            msg = Msg(pkt=pkt, src_ip=src_ip)
            if ( self.subsystem == 'ALL' or
                 msg.dst        == 'ALL' or
                 self.subsystem == msg.dst ):
                self.msg_queue.put(msg)
    def shutdown(self):
        self.msg_queue.put(ConsumerThread.STOP)
        #print(self.name, "shutdown")
    def get(self, timeout=None):
        try:
            return self.msg_queue.get(True, timeout)
        except queue.Empty:
            return None

class MsgSender(ConsumerThread):
    def __init__(self, dst_addr, subsystem,
                max_attempts=5):
        ConsumerThread.__init__(self)
        self.subsystem    = subsystem
        self.max_attempts = max_attempts
        self.socket       = socket.socket(socket.AF_INET,
                                        socket.SOCK_DGRAM)
        #self.socket.connect(address)
        self.dst_ip   = dst_addr[0]
        self.dst_port = dst_addr[1]
        self.name = 'MCS.MsgSender'
    def process(self, msg):
        msg.src  = self.subsystem
        try:
            pkt      = msg.encode()
        except UnicodeDecodeError:
            pkt      = msg.data
        dst_ip   = msg.dst_ip if msg.dst_ip is not None else self.dst_ip
        dst_addr = (dst_ip, self.dst_port)
        #print("Sending msg to", dst_addr)
        for attempt in range(self.max_attempts-1):
            try:
                #self.socket.send(pkt)
                self.socket.sendto(pkt, dst_addr)
            except socket.error:
                time.sleep(0.001)
            else:
                return
        #self.socket.send(pkt)
        self.socket.sendto(pkt, dst_addr)
    def shutdown(self):
        #print(self.name, "shutdown")
        pass

# Simple interface for communicating with adp-control service
class Communicator(object):
    def __init__(self, subsystem='MCS'):
        # TODO: Load port numbers etc. from config
        #sender   = MsgSender(("localhost",1742), subsystem=subsystem)
        self.sender   = MsgSender(("adp",1742), subsystem=subsystem)
        self.receiver = MsgReceiver(("0.0.0.0",1743))
        self.sender.input_queue = queue.Queue()
        self.sender.daemon = True
        self.receiver.daemon = True
        self.sender.start()
        self.receiver.start()
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.sender.request_stop()
        self.receiver.request_stop()
        self.sender.join()
        self.receiver.join()
    def _get_reply(self, timeout):
        reply = self.receiver.get(timeout=timeout)
        if reply is None:
            raise RuntimeError("MCS request timed out")
        # Parse the data section of the reply
        response, status, data = reply.data[:1], reply.data[1:8], reply.data[8:]
        try:
            response = response.decode()
            status = status.decode()
        except AttributeError:
            # Python2 catch
            pass
        if response != 'A':
            raise ValueError("Message not accepted: response=%r, status=%r, data=%r" % (response, status, data))
        return status, data
    def _send(self, msg, timeout):
        self.sender.put(msg)
        self.status, data = self._get_reply(timeout)
        return data
    def report(self, data, fmt=None, timeout=5.):
        msg = Msg(dst='ADP', cmd='RPT', data=data)
        data = self._send(msg, timeout)
        if fmt is None or fmt == 's':
            try:
                data = data.decode()
            except AttributeError:
                # Python2 catch
                pass
            return data
        else:
            return struct.unpack('>'+fmt, data)
    def command(self, cmd, data='', timeout=5.):
        msg = Msg(dst='ADP', cmd=cmd, data=data)
        self._send(msg, timeout)

class SafeSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        socket.socket.__init__(self, *args, **kwargs)
        l_onoff = 1
        l_linger = 0
        self.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                        struct.pack('ii', l_onoff, l_linger))

class Synchronizer(object):
    def __init__(self, group, addr=("adp",23820)):
        self.group  = group
        self.addr   = addr
        self.socket = SafeSocket(socket.AF_INET,
                                socket.SOCK_STREAM)
        self.socket.connect(addr)
        self.socket.settimeout(10) # Prevent recv() from blocking indefinitely
        msg = 'GROUP:'+str(group)
        try:
            msg = msg.encode()
        except AttributeError:
            # Python2 catch
            pass
        self.socket.send(msg)
    def __call__(self, tag=None):
        msg = 'TAG:'+str(tag)
        try:
            msg = msg.encode()
        except AttributeError:
            # Python2 catch
            pass
        self.socket.send(msg)
        reply = self.socket.recv(4096)
        try:
            reply = reply.decode()
        except AttributeError:
            # Python2 catch
            pass
        expected_reply = 'GROUP:'+str(self.group) + ',TAG:'+str(tag)
        if reply != expected_reply:
            raise ValueError("Unexpected reply '%s', expected '%s'" %
                            (reply, expected_reply))

class SynchronizerGroup(object):
    def __init__(self, group):
        self.group = group
        self.socks = []
        self.pending_lock = Semaphore()
        self.shutdown_event = Event()
        self.run_thread = Thread(target=self.run)
        #self.run_thread.daemon = True
        self.tStart = time.time()
        self.run_thread.start()
    #def __del__(self):
    #    self.shutdown()
    def log(self, value):
        print("[%.3f] %s" % (time.time()-self.tStart, value))
    def shutdown(self):
        self.shutdown_event.set()
        self.log("SynchronizerGroup "+self.group+": run joining")
        self.run_thread.join()
        self.log("SynchronizerGroup "+self.group+": run thread joined")
    def add(self, sock, addr):
        try:
            self.pending_lock.acquire()
            self.pending.append(sock)
        except AttributeError:
            self.pending = [sock,]
        finally:
            self.pending_lock.release()
    def run(self):
        while not self.shutdown_event.is_set():
            try:
                self.pending_lock.acquire()
                self.socks.extend(self.pending)
                self.log("SynchronizerGroup "+self.group+": added "+str(len(self.pending))+" clients")
                del self.pending
            except AttributeError:
                pass
            finally:
                self.pending_lock.release()

            if len(self.socks) == 0:
                # Avoid spinning after all socks close
                time.sleep(0.1)
                continue

            # Find out where everyone is
            tags = []
            i = 0
            while i < len(self.socks) and not self.shutdown_event.is_set():
                sock = self.socks[i]
                try:
                    tag_msg = sock.recv(4096)
                except socket.timeout as e:
                    self.log("WARNING: Synchronizer (1a): socket.timeout %s client %i: %s" % (self.group, i, e))
                    self.socks[i].close()
                    del self.socks[i]
                    continue
                except socket.error as e:
                    self.log("WARNING: Synchronizer (1b): socket.error %s client %i: %s" % (self.group, i, e))
                    self.socks[i].close()
                    del self.socks[i]
                    continue
                if tag_msg[:4] != 'TAG:':
                    e = tag_msg
                    self.log("WARNING: Synchronizer (1c): Unexpected message %s client %i: %s" % (self.group, i, e))
                    self.socks[i].close()
                    del self.socks[i]
                    continue
                tags.append( int(tag_msg[4:22], 10) )
                i += 1

            # Elect tag0, the reference time tag
            try:
                tag0 = max(tags)
                #print("ELECTED %i as tag0 for %s" % (tag0, self.group))
            except ValueError:
                continue

            # Speed up the slow ones a little bit
            slow = [i for i,tag in enumerate(tags) if tag < tag0]
            if len(slow) > 0:
                j = 0
                #slowFactors = {}
                while slow and j < 5 and not self.shutdown_event.is_set():
                    ## Deal with each slow client in turn
                    for i in slow:
                        ### Send - ignoring errors
                        sock, tag = self.socks[i], tags[i]
                        try:
                            sock.send('GROUP:'+self.group+',TAG:'+str(tag0))
                            #try:
                            #    slowFactors[i] += 1
                            #except KeyError:
                            #    slowFactors[i] = 1
                        except socket.error as e:
                            self.log("WARNING: Synchronizer (2a): socket.error %s client %i: %s" % (self.group, i, e))

                        ### Receive - ignoring errors
                        try:
                            tag_msg = sock.recv(4096)
                        except socket.timeout as e:
                            self.log("WARNING: Synchronizer (2b): socket.timeout %s client %i: %s" % (self.group, i, e))
                        except socket.error as e:
                            self.log("WARNING: Synchronizer (2c): socket.error %s client %i: %s" % (self.group, i, e))
                        if not tag_msg.startswith('TAG:'):
                            e = tag_msg
                            self.log("WARNING: Synchronizer (2d): Unexpected message %s client %i: %s" % (self.group, i, e))
                            continue
                        tags[i] = int(tag_msg[4:22], 10)
                        #print("Updated %s client %i tag to %i (tag0 is %i; delta is now %i" % (self.group, i, tags[i], tag0, tags[i]-tag0))

                    ## Evaluate the latest batch of timetags
                    slow = [i for i,tag in enumerate(tags) if tag < tag0]

                    ## Update the iteration variable
                    j += 1

                ### Report on what we've done
                #for i,v in slowFactors.items():
                #	print("WARNING: Synchronizer (2e): slipped %s client %i forward by %s" % (self.group, i, v))

            # Send to everyone regardless to make sure the fast ones don't falter
            i = 0
            while i < len(self.socks):
                sock, tag = self.socks[i], tags[i]
                if tag != tag0:
                    self.log("WARNING: Synchronizer (3a): Tag mismatch: "+str(tag)+" != "+str(tag0)+" from "+self.group+" client "+str(i)+" (delta is "+str((tag-tag0)/196e6*1000)+" ms)")

                try:
                    sock.send('GROUP:'+self.group+',TAG:'+str(tag0))
                except socket.error as e:
                    self.log("WARNING: Synchronizer (3b): socket.error: %s client %i: %s" % (self.group, i, e))
                    self.socks[i].close()
                    del self.socks[i]
                    del tags[i]
                    continue
                i += 1

            # Done with the iteration
            ##print("SYNCED "+str(len(self.socks))+" clients in "+self.group)

        self.log("SynchronizerGroup "+self.group+": shut down")

class SynchronizerServer(object):
    def __init__(self, addr=("0.0.0.0",23820)):
        self.addr = addr
        self.sock = SafeSocket(socket.AF_INET,
                                socket.SOCK_STREAM)
        self.sock.settimeout(5) # Prevent accept() from blocking indefinitely
        self.sock.bind(addr)
        self.sock.listen(32)
        self.groups = {}#defaultdict(SynchronizerGroup)
        self.shutdown_event = Event()
    def shutdown(self):
        self.shutdown_event.set()
    def run(self):
        while not self.shutdown_event.is_set():
            try:
                sock, addr = self.sock.accept()
                sock.settimeout(10)
            except socket.timeout:
                continue
            group_msg = sock.recv(4096)
            if not group_msg.startswith('GROUP:'):
                #raise ValueError("Unexpected message: "+group_msg)
                print("WARNING: Synchronizer: Unexpected message: "+group_msg)
            group = group_msg[len('GROUP:'):]
            if group not in self.groups:
                self.groups[group] = SynchronizerGroup(group)
            self.groups[group].add(sock, addr)
        for group in self.groups.values():
            group.shutdown()
        # Note: This seems to be necessary to avoid 'address already in use'
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()

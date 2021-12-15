# -*- coding: utf-8 -*-

from __future__ import print_function

import threading

class ConsumerThread(threading.Thread):
    class StopObject(object): pass
    STOP = StopObject
    def __init__(self, input_queue=None):
        threading.Thread.__init__(self)
        self.input_queue = input_queue
    def request_stop(self):
        # Note: If multiple consumers for one queue, calling this on each
        #         should still work to stop them all, just not in any order.
        #         Must be careful though to call all request_stop()'s before
        #           calling join()'s.
        self.input_queue.put(ConsumerThread.STOP)
    def run(self):
        while True:
            task = self.input_queue.get()
            if task is ConsumerThread.STOP:
                self.input_queue.task_done()
                break
            try:
                self.process(task)
            except Exception as e:
                print("ERROR: Uncaught exception in %s: %s" % (self, e))
            self.input_queue.task_done()
        self.shutdown()
    def process(self, task):
        # Implement this in the subclass
        pass
    def shutdown(self):
        # Implement this in the subclass
        pass
    # Handy method for testing
    def put(self, task, timeout=None):
        self.input_queue.put(task, timeout)

if __name__ == "__main__":
    from Queue import Queue
    import time
    
    class MyConsumer(ConsumerThread):
        def process(self, task):
            #time.sleep(5)
            print(task)
            
    q = Queue()
    m = MyConsumer(q)
    m.daemon = True
    m.start()
    q.put('Hello')
    time.sleep(0.2)
    q.put('world!')
    time.sleep(0.2)
    m.request_stop()

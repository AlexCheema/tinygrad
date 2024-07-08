import argparse, pickle, socket, ctypes
from socketserver import BaseRequestHandler, TCPServer
from tinygrad import Device

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--port", type=int, default=1234)
  args = parser.parse_args()

  class RemoteHandler(BaseRequestHandler):
    def __init__(self, *args, **kwargs):
      self.device = Device[Device.DEFAULT]

      super().__init__(*args, **kwargs)

    def handle(self):
      programs = {}
      buffers = {}
      allocators_dict = {}

      while True:
        cmd = self.request.recv(1)
        match cmd:
          case b"\x00": # device
            device = self.request.recv(1024).decode("utf-8")
            self.device = Device[device]
            self.request.send(b"\x00")
          case b"\x01": # synchronize
            self.device.synchronize()
            self.request.send(b"\x00")
          case b"\x02": # allocate
            size, options = pickle.loads(self.request.recv(1024))
            opaque = self.device.allocator.alloc(size, options)
            allocators_dict[hash(opaque)] = opaque
            buffers[hash(opaque)] = bytearray(size)
            pickled = pickle.dumps(hash(opaque))
            self.request.sendall(pickled)
          case b"\x03": # free
            opaque, options = pickle.loads(self.request.recv(1024))
            del buffers[opaque]
            self.device.allocator.free(allocators_dict[opaque], 0, options)
            self.request.send(b"\x00")
          case b"\x04": # copyin
            dest = ctypes.c_ulong(int.from_bytes(self.request.recv(8), "little"))
            src = memoryview(buffers[dest.value])
            total = 0
            while total < src.nbytes:
              recv = self.request.recv_into(src[total:], src.nbytes - total)
              total += recv
            self.device.allocator.copyin(allocators_dict[dest.value], src)
          case b"\x05": # copyout
            src = ctypes.c_ulong(int.from_bytes(self.request.recv(8), "little"))
            dest = buffers[src.value]
            self.device.allocator.copyout(memoryview(dest), allocators_dict[src.value])
            self.request.sendall(dest)
          case b"\x06": # compile
            nbytes = int.from_bytes(self.request.recv(4), "little")
            src_bytes = bytearray(nbytes)
            src_view = memoryview(src_bytes)
            total = 0
            while total < nbytes:
              recv = self.request.recv_into(src_view[total:], nbytes - total)
              total += recv
            src = src_bytes.decode("utf-8")
            compiled = self.device.compiler.compile(src)
            self.request.sendall(len(compiled).to_bytes(4, "little") + compiled)
          case b"\x07": # load
            name, nbytes, iden = pickle.loads(self.request.recv(1024))
            self.request.send(b"\x00")
            lib = bytearray(nbytes)
            lib_view = memoryview(lib)
            total = 0
            while total < nbytes:
              recv = self.request.recv_into(lib_view[total:], nbytes - total)
              total += recv
            programs[iden] = self.device.runtime(name, bytes(lib))
            self.request.send(b"\x00")
          case b"\x08": # run
            name, bufs, global_size, local_size, vals, wait, iden = pickle.loads(self.request.recv(4096))
            bufs = [allocators_dict[buf] for buf in bufs]
            try: programs[iden](*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)
            except: failed = 1
            else: failed = 0
            self.request.send(bytes([failed]))
          case b"\xff": # exit
            print("exit")
            break
          case b"": break
          case _: print(f"Unknown {cmd=}")
      self.request.close()

  server = TCPServer(("0.0.0.0", args.port), RemoteHandler)
  server.allow_reuse_address = True
  server.allow_reuse_port = True
  server.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  with server: server.serve_forever()

from typing import Tuple, Optional
import pickle, time, socket
from tinygrad.device import Compiled, Compiler, Allocator
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer, HIPRenderer, OpenCLRenderer, MetalRenderer
from tinygrad.codegen.uops import UOpGraph

def RemoteProgram(s: socket.socket):
  class _RemoteProgram:
    def __init__(self, name:str, lib:bytes):
      self.name, self.lib = name, lib
      s.sendall(b"\x07")
      s.sendall(pickle.dumps((name, len(self.lib), id(self))))
      s.recv(1)
      s.sendall(self.lib)
      s.recv(1)
    def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
      st = time.perf_counter()
      s.sendall(b"\x08" + pickle.dumps((self.name, bufs, global_size, local_size, vals, wait, id(self))))
      if s.recv(1) == b"\x01": raise RuntimeError("kernel failed")
      return time.perf_counter() - st
  return _RemoteProgram

class RemoteRenderer(Renderer):
  def __init__(self, device:str, rdevice:str):
    self.device, self.rdevice = device, rdevice
    if rdevice.startswith("CUDA"): self.rrenderer = CUDARenderer("sm_89")
    elif rdevice.startswith("HSA"): self.rrenderer = HIPRenderer()
    elif rdevice.startswith("GPU"): self.rrenderer = OpenCLRenderer()
    elif rdevice.startswith("METAL"): self.rrenderer = MetalRenderer()
    else: raise ValueError(f"Unsupported device {device}")
  def render(self, name:str, uops:UOpGraph) -> str: return self.rrenderer.render(name, uops)

class RemoteCompiler(Compiler):
  def __init__(self, s: socket.socket, cachekey:Optional[str]=None):
    self.s = s
    super().__init__(cachekey)
  def compile(self, src:str) -> bytes:
    src_bytes = src.encode("utf-8")
    self.s.sendall(b"\x06" + len(src_bytes).to_bytes(4, "little") + src_bytes)
    nbytes = int.from_bytes(self.s.recv(4), "little")
    lib = bytearray(nbytes)
    lib_view = memoryview(lib)
    total = 0
    while total < nbytes:
      recv = self.s.recv_into(lib_view[total:], nbytes - total)
      total += recv
    return bytes(lib)

class RemoteAllocator(Allocator):
  def __init__(self, s: socket.socket): self.s = s
  def _alloc(self, size, options):
    self.s.sendall(b"\x02" + pickle.dumps((size, options)))
    recv = self.s.recv(1024)
    return pickle.loads(recv)
  def _free(self, opaque, options):
    try: args = pickle.dumps((opaque, options))
    except: return
    self.s.sendall(b"\x03" + args)
    self.s.recv(1)
  def copyin(self, dest, src:memoryview):
    self.s.sendall(b"\x04" + int(dest).to_bytes(8, "little") + src.tobytes())
  def copyout(self, dest:memoryview, src):
    self.s.sendall(b"\x05" + int(src).to_bytes(8, "little"))
    total = 0
    while total < dest.nbytes:
      recv = self.s.recv_into(dest[total:], dest.nbytes - total)
      total += recv

class RemoteDevice(Compiled):
  def __init__(self, odevice:str):
    device = odevice.lstrip("REMOTE:")
    ip = device.split(":")[0]
    port = device.split(":")[1]
    rdevice = device[len(ip)+1+len(port)+1:]
    print("connecting to socket", ip, port, rdevice)

    # open up socket to remote device
    self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    self.s.connect((ip, int(port)))
    # send device
    self.s.sendall(b"\x00" + rdevice.encode("utf-8"))
    self.s.recv(1)

    super().__init__(odevice, RemoteAllocator(self.s), RemoteRenderer(odevice, rdevice), RemoteCompiler(self.s), RemoteProgram(self.s))
  def __del__(self):
    self.s.send(b"\xff")
    self.s.close()

  def synchronize(self):
    self.s.send(b"\x01")
    self.s.recv(1)

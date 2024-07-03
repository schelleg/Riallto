from sys import path
path.append('C:\\Users\\grahams\\notebooks\\Riallto.schelleg')


from npu.build.appbuilder import AppBuilder
from npu.build.kernel import Kernel
from npu.runtime import AppRunner
import numpy as np



class SingleKernelApp(AppBuilder):
    def __init__(self, kernelfx, inputargs, outputargs):
        self.kernelfx = kernelfx
        self.inputargs = inputargs
        self.args = inputargs + outputargs
        self.bufs = [b for b in self.args if isinstance(b, np.ndarray)]
        self.app = None
        super().__init__()

    def callgraph(self, *args, **kwargs):
        return self.kernelfx(*args, **kwargs)

    def build(self):
        super().build(*self.inputargs)
    
    def run(self):
        if self.app is None:
            self.app = AppRunner('SingleKernelApp.xclbin') 

        appbufs = [self.app.allocate(shape=b.shape, dtype=b.dtype) for b in self.bufs]
        _ = [np.copyto(appbufs[ix], b) for ix,b in enumerate(self.bufs)]
        _ = [b.sync_to_npu() for b in appbufs]

        self.app.call(*appbufs)

        _ = [b.sync_from_npu() for b in appbufs]
        _ = [print(b) for b in appbufs]  


def vbc_behavioral(obj):
    obj.c = obj.a
    obj.b = obj.a

vectorbroadcast = Kernel('''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

extern "C" {
  void vectorbroadcast(uint8_t* a, uint8_t* b, uint8_t* c, const uint32_t nbytes) {

    ::aie::vector<uint8_t, 32> ai, bi, ci;

    for(int j=0; j<nbytes; j+=32) {
        ai = ::aie::load_v<32>(a);
        a += 32;
        ::aie::store_v(b, ai);
        b += 32;
        ::aie::store_v(c, ai);
        c += 32;
    }
 }
}
''',vbc_behavioral)


a = np.random.randint(0, 256, size=4096, dtype=np.uint8)
b = np.zeros(4096, dtype=np.uint8)
c = np.zeros(4096, dtype=np.uint8)


vectorbroadcast.b.array = b   # So we don't have to write a behavioral model, just set the C array dims
vectorbroadcast.c.array = c   # So we don't have to write a behavioral model, just set the C array dims
vectorbroadcastapp = SingleKernelApp(vectorbroadcast, inputargs=[a, a.nbytes], outputargs=[b, c])


vectorbroadcastapp.build()
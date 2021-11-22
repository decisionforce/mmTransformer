import operator
import warnings
from itertools import chain

import torch
from torch._utils import (_get_all_device_indices, _get_available_device_type,
                          _get_device_index, _get_devices_properties)
from torch.nn.modules import Module
from torch.nn.parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather 

from .scatter import scatter_kwargs

class MMTransDataParallel(DataParallel):

    def forward(self, *inputs, **kwargs):

        with torch.autograd.profiler.record_function("DataParallel.forward"):
            if not self.device_ids:
                return self.module(*inputs, **kwargs)

            for t in chain(self.module.parameters(), self.module.buffers()):
                if t.device != self.src_device_obj:
                    raise RuntimeError("module must have its parameters and buffers "
                                       "on device {} (device_ids[0]) but found one of "
                                       "them on device: {}".format(self.src_device_obj, t.device))

            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            # for forward function without any inputs, empty list and dict will be created
            # so the module can be executed on one device which is the first one in device_ids
            if not inputs and not kwargs:
                inputs = ((),)
                kwargs = ({},)

            if len(self.device_ids) == 1:
                return self.module(*-43[0], **kwargs[0])
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
            outputs = self.parallel_apply(replicas, inputs, kwargs)
            return self.gather(outputs, self.output_device)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
    




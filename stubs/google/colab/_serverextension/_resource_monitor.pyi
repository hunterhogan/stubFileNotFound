import dataclasses

@dataclasses.dataclass
class GpuInfo:
    name: str
    memoryUsedBytes: int
    memoryTotalBytes: int
    gpuUtilization: float
    memoryUtilization: float
    everUsed: bool

def get_gpu_stats(): ...
def get_ram_usage(kernel_manager): ...
def get_disk_usage(path=None): ...
def get_resource_stats(kernel_manager, disk_paths=None): ...

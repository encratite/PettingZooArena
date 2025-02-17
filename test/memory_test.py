import numpy as np
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

SHARED_MEMORY_ID = "test"

def entry_point(count):
	process_memory = SharedMemory(SHARED_MEMORY_ID)
	process_array = np.frombuffer(process_memory.buf, dtype=np.int8, count=count)
	print(process_array)
	del process_array
	process_memory.close()

if __name__ == "__main__":
	data = [1, 3, 7, 9, 9]
	size = len(data) * np.dtype(np.int8).itemsize
	memory = SharedMemory(SHARED_MEMORY_ID, size=size, create=True)
	array = np.ndarray((size,), dtype=np.int8, buffer=memory.buf)
	array[:] = data
	process = Process(target=entry_point, args=(len(data),))
	process.start()
	process.join()
	memory.close()
	memory.unlink()
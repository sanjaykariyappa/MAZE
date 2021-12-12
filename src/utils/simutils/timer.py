import time
def timer(fun, *args):
    start = time.time()
    fun(*args)
    end = time.time()
    elapsed = end - start
    print("Total Runtime= " + str(int(elapsed / (60 * 60))) + ":" + str(int((elapsed / 60) % 60)) + ":" + str(
        int(elapsed % 60)) + '\n')


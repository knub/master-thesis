import resource
import logging

def limit_memory(memory_in_mb):
    rsrc = resource.RLIMIT_AS
    soft, hard = resource.getrlimit(rsrc)
    logging.info("Setting memory limit, previous: (%s, %s), now: (%s, %s)" %
                 (_byte_to_mb(soft), _byte_to_mb(hard), memory_in_mb, _byte_to_mb(hard)))

    resource.setrlimit(rsrc, (_mb_to_byte(memory_in_mb), hard))

def _mb_to_byte(mb):
    return mb * 1024 * 1024
def _byte_to_mb(bytes):
    return bytes / (1024 * 1024)

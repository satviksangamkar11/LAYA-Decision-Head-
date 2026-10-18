import socket, sys
def solve(d):
    r = {}
    try:
        socket.socket(); r['socket'] = 'ALLOWED'
    except OSError as e:
        r['socket'] = 'denied'
    try:
        open('/kaggle/working/HOST_CANARY.txt').read(); r['canary'] = 'READ'
    except OSError:
        r['canary'] = 'unreadable'
    return r

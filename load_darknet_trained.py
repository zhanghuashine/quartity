import struct


def read_darknet_trained_head(f):
    '''
    note: reference to darknet, get the head of **.weights
    :param f: f = open('', 'rb')
    :return: list
    '''
    major = struct.unpack('i', f.read(4))[0]
    minor = struct.unpack('i', f.read(4))[0]
    revision = struct.unpack('i', f.read(4))[0]
    if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
        seen = struct.unpack('l', f.read(8))[0]
    else:
        seen = struct.unpack('i', f.read(4))[0]
    return [major, minor, revision, seen]

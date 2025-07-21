import data

def lpDecoder(license, char=data.CHARS):
    
    decoded=[]
    print(license)
    for idx in license:
        if char[idx]!= '-':
            decoded.append(char[idx])
    return ''.join(decoded)
    

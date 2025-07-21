import data

def lpDecoder(license,pr_list=data.provinces, char_list=data.ads):
    
    decoded=[]
    print(license)
    for idx in license.squeeze():
        if(len(decoded)==0):
            decoded.append(pr_list[idx])
        else:
            decoded.append(char_list[idx])
    return ''.join(decoded)
    

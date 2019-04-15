def reading_dataset(file_name, n, d):
    out = [0] * n
    for i in range(n):
        out[i] = [0] * (d+1)
    f = open(file_name, 'r')
    for i in range(n):
        s=f.readline()
        tokens=s.split(' ')
        out[i][0]=tokens[0]
        out[i][0]=-3+2*out[i][0]
        del tokens[0]
        while len(tokens)>1:
            print(tokens)
            num_and_val=tokens[0].split(':')
            out[i][ int(num_and_val[0]) ]= num_and_val[1]
            del tokens[0]
    return out

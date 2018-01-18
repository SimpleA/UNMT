import sys

out_file = open(sys.argv[1]+'.proc', 'w', encoding='utf-8')
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    for l in f:
        l = l.strip()
        words = l.split(' ')
        cont = []
        for w in words:
            if w != '<BOS>' and w != '<PAD>' and w != '<EOS>':
                cont.append(w)
            if w == '<EOS>':
                break

        out_file.write(' '.join(cont) + '\n')
    

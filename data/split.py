from sys import stdin
out = None
nametemplate = 'csa/out_{:010d}.csa'
i = 0

for line in stdin:
    line = line.rstrip()
    if line == 'V2.2':
        if out is not None:
            out.close()

        out = open(nametemplate.format(i), mode='w')
        i = i + 1

    if out is not None:
        print(line, file=out)


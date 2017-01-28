from pathlib import Path

def load_file(path):
    f = Path(path)

    lines = f.open().readlines()
    lines = list(map(lambda x: x.rstrip(), lines))
    lines = filter(lambda x: len(x) > 0, lines)

    return list(lines)

def load_files(path):
    d = Path(path)

    records = []

    for f in d.glob('*.csa'):
        lines = f.open().readlines()
        lines = list(map(lambda x: x.rstrip(), lines))
        lines = filter(lambda x: len(x) > 0, lines)
        records.extend(lines)

    return records

def to_data(r):
    sfen, total, move, winner = r.split(',')
    side = sfen.split(' ')[1]
    turn = sfen.split(' ')[3]
    return sfen, side, turn, total, move, winner


from argparse import ArgumentParser
from pathlib import Path
import tensorflow as tf
pio = tf.python_io

import ginkgo.shogi_records as sr
import ginkgo.numpy_shogi as ns

def create_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None)
    parser.add_argument('-o', '--output-dir', type=str, default='./')
    return parser

def feature_int64(a):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[a]))

def feature_bytes(a):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[a]))

def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    proc(args)

def proc(args):
    input_path = Path(args.input)
    convert_file(args, input_path)

def convert_file(args, file_path):
    records = list(map(sr.to_data, sr.load_file(file_path)))
    output_file = Path(args.output_dir) / (file_path.stem + '.tfrecords')
    writer = pio.TFRecordWriter(
        str(output_file), options=pio.TFRecordOptions(pio.TFRecordCompressionType.GZIP))

    for r in records:
        # record tuple
        # 0 .. sfen
        # 1 .. side (first black, second white)
        # 2 .. turn number
        # 3 .. total turn number
        # 4 .. next movement
        # 5 .. winner side
        vec = ns.sfen_to_vector(r[0], usi=r[4])

        if r[1] == 'w':
            vec = ns.player_inverse(vec)

        #print(type(vec))
        print(vec.dtype)

        record = tf.train.Example(features=tf.train.Features(feature={
            'vec': feature_bytes(vec.tostring()),
            'turn_number' : feature_int64(int(r[2])),
            'total_turn_number' : feature_int64(int(r[3])),
            'side' : feature_int64(0 if r[1] == 'b' else 1),
            'winner_side' : feature_int64(0 if r[5] == 'b' else 1),
            }))
        writer.write(record.SerializeToString())

    writer.close()

if __name__ == '__main__':
    main()


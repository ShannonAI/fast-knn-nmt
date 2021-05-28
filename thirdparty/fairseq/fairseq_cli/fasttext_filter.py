# encoding: utf-8
"""



@desc: 

"""

from argparse import ArgumentParser
import fasttext
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--tgt", type=str, required=True)
    parser.add_argument("--src_out", type=str, required=True)
    parser.add_argument("--tgt_out", type=str, required=True)
    parser.add_argument("--src_lang", type=str, default="en")
    parser.add_argument("--tgt_lang", type=str, default="de")
    args = parser.parse_args()
    total = 0
    valid = 0
    model = fasttext.load_model(args.model)
    with open(args.src) as fsrc, open(args.tgt) as ftgt, \
        open(args.src_out, "w") as fsrc_out, open(args.tgt_out, "w") as ftgt_out:
        for s, t in tqdm(zip(fsrc, ftgt)):
            total += 1
            s = s.strip()
            t = t.strip()
            if not s or not t:
                continue
            slang = model.predict([s])[0][0][0]
            tlang = model.predict([t])[0][0][0]
            if slang != "__label__" + args.src_lang or tlang != "__label__" + args.tgt_lang:
                continue
            fsrc_out.write(s+"\n")
            ftgt_out.write(t+"\n")
            valid += 1
    print(f"Wrote {valid}/{total} lines to {args.src_out} and {args.tgt_out}")


if __name__ == '__main__':
    main()

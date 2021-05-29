# encoding: utf-8
"""



@desc: compute ppl according to file generated by `fairseq-generate --score-reference`

"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    args = parser.parse_args()
    total_score = 0
    total_count = 0
    with open(args.file) as fin:
        for line in fin:
            line = line.strip()
            if not line.startswith("P"):
                continue
            scores = line.split("\t")[1]
            scores = [float(x) for x in scores.split()]
            for s in scores:
                total_score += s
                total_count += 1
    print(f"total score: {total_score}, total words: {total_count}, average score: {total_score/total_count}")


if __name__ == '__main__':
    main()
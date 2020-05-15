import glob
import pickle
import jsonlines

if __name__ == "__main__":
    files = glob.glob("data/codesearchnet_javascript/augmented/*.jsonl*.augmented")
    files.sort()
    all_data = []
    for fname in files:
        with open(fname, "r") as readf:
            reader = jsonlines.Reader(readf)
            for line in reader:
                alternatives = set(line)
                print(len(alternatives))
                all_data.append(alternatives)
        print("Finished processing", fname)
    with open("data/codesearchnet_javascript/augmented/data.pickle", "wb") as writef:
        pickle.dump(all_data, writef, protocol=pickle.HIGHEST_PROTOCOL)
    print("Wrote pickle")

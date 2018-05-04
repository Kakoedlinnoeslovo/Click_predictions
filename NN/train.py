from utils import ReaderSubmitor

def main():
    reader = ReaderSubmitor()
    all_values = reader.get_values()
    chunk = reader.next_chunk(all_values = all_values, chunksize=100)


if __name__ == '__main__':
    main()
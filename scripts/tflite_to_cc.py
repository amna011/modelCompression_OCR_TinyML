import argparse


def convert(in_path: str, out_path: str, var_name: str) -> None:
    with open(in_path, "rb") as f:
        data = f.read()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated from %s\n" % in_path)
        f.write("#include <stdint.h>\n\n")
        f.write(f"const unsigned char {var_name}[] = {{\n")
        for i, b in enumerate(data):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{b:02x}, ")
            if i % 12 == 11:
                f.write("\n")
        if len(data) % 12 != 0:
            f.write("\n")
        f.write("};\n")
        f.write(f"const unsigned int {var_name}_len = {len(data)};\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TFLite model to C array.")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--var", dest="var_name", default="g_model")
    args = parser.parse_args()

    convert(args.in_path, args.out_path, args.var_name)


if __name__ == "__main__":
    main()

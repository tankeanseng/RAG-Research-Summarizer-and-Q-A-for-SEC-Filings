import os
import zipfile

SRC_DIR = "build"
OUT_ZIP = "package.zip"

def main():
    if os.path.exists(OUT_ZIP):
        os.remove(OUT_ZIP)

    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(SRC_DIR):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, SRC_DIR)
                z.write(full_path, rel_path)

    print(f"Created {OUT_ZIP}")

if __name__ == "__main__":
    main()
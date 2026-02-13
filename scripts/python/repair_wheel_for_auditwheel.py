#!/usr/bin/env python3

import argparse
import base64
import csv
import hashlib
import pathlib
import shutil
import tempfile
import zipfile


def update_wheel_metadata(root: pathlib.Path) -> None:
    wheel_files = list(root.glob("*.dist-info/WHEEL"))
    if not wheel_files:
        raise RuntimeError("Cannot find WHEEL metadata file.")

    wheel_file = wheel_files[0]
    text = wheel_file.read_text(encoding="utf-8")
    text = text.replace("Root-Is-Purelib: true", "Root-Is-Purelib: false")
    wheel_file.write_text(text, encoding="utf-8")


def move_purelib_to_platlib(root: pathlib.Path) -> None:
    for purelib_dir in root.glob("*.data/purelib"):
        platlib_dir = purelib_dir.parent / "platlib"
        platlib_dir.mkdir(parents=True, exist_ok=True)

        for source in sorted(purelib_dir.rglob("*")):
            if source.is_dir():
                continue
            relative = source.relative_to(purelib_dir)
            target = platlib_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))

        shutil.rmtree(purelib_dir)


def file_digest_and_size(path: pathlib.Path) -> tuple[str, int]:
    data = path.read_bytes()
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"sha256={encoded}", len(data)


def rewrite_record(root: pathlib.Path) -> None:
    dist_info_dirs = list(root.glob("*.dist-info"))
    if not dist_info_dirs:
        raise RuntimeError("Cannot find .dist-info directory.")

    record_path = dist_info_dirs[0] / "RECORD"
    if not record_path.exists():
        raise RuntimeError("Cannot find RECORD file.")

    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.endswith(".dist-info/RECORD"):
            rows.append((rel, "", ""))
            continue
        digest, size = file_digest_and_size(path)
        rows.append((rel, digest, str(size)))

    with record_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def repack_wheel(src_root: pathlib.Path, output_wheel: pathlib.Path) -> None:
    with zipfile.ZipFile(output_wheel, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_root.rglob("*")):
            if path.is_dir():
                continue
            zf.write(path, path.relative_to(src_root).as_posix())


def prepare_wheel(wheel_path: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = pathlib.Path(temp_dir_str)
        unpack_root = temp_dir / "wheel"
        unpack_root.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(wheel_path, "r") as zf:
            zf.extractall(unpack_root)

        update_wheel_metadata(unpack_root)
        move_purelib_to_platlib(unpack_root)
        rewrite_record(unpack_root)

        rebuilt = temp_dir / wheel_path.name
        repack_wheel(unpack_root, rebuilt)
        shutil.copyfile(rebuilt, wheel_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=pathlib.Path)
    args = parser.parse_args()

    wheel_path = args.wheel.resolve()
    if not wheel_path.exists():
        raise FileNotFoundError(f"Wheel not found: {wheel_path}")

    prepare_wheel(wheel_path)
    print(f"Prepared wheel for auditwheel: {wheel_path}")


if __name__ == "__main__":
    main()

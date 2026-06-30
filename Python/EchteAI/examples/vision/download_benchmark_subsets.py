import argparse
import json
import shutil
import zipfile
from pathlib import Path

import requests
from requests.exceptions import SSLError


KITTI_IMAGE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
KITTI_LABEL_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
COCO_ANNOTATIONS_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_BASE_URL = "https://images.cocodataset.org/val2017"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download small benchmark subsets for KITTI and COCO."
    )
    parser.add_argument(
        "--dataset",
        choices=["kitti", "coco", "all"],
        default="all",
        help="Which subset to prepare.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="How many images to keep for each subset.",
    )
    parser.add_argument(
        "--output-root",
        default="./downloads/benchmark_subsets",
        help="Where the subsets will be written.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path, timeout: int, expect_zip: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not expect_zip or zipfile.is_zipfile(destination):
            return destination
        destination.unlink()

    print(f"Downloading: {url}")
    try:
        response = requests.get(url, stream=True, timeout=timeout)
    except SSLError:
        print(f"SSL verification failed for {url}, retrying with verify=False")
        response = requests.get(url, stream=True, timeout=timeout, verify=False)

    with response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    if expect_zip and not zipfile.is_zipfile(destination):
        destination.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file is not a valid zip archive: {destination}")

        return destination
    return destination


def extract_selected_members(zip_path: Path, members, destination_root: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in members:
            target_path = destination_root / member
            if target_path.exists():
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def build_kitti_subset(output_root: Path, limit: int, timeout: int) -> Path:
    subset_root = output_root / "kitti"
    archive_root = output_root / "_archives"
    image_zip = download_file(
        KITTI_IMAGE_URL, archive_root / "data_object_image_2.zip", timeout, expect_zip=True
    )
    label_zip = download_file(
        KITTI_LABEL_URL, archive_root / "data_object_label_2.zip", timeout, expect_zip=True
    )

    with zipfile.ZipFile(image_zip, "r") as image_archive:
        image_members = sorted(
            name
            for name in image_archive.namelist()
            if name.startswith("training/image_2/") and name.lower().endswith(".png")
        )[:limit]

    label_members = [
        f"training/label_2/{Path(image_name).stem}.txt"
        for image_name in image_members
    ]

    extract_selected_members(image_zip, image_members, subset_root)
    extract_selected_members(label_zip, label_members, subset_root)

    (subset_root / "testing" / "image_2").mkdir(parents=True, exist_ok=True)
    (subset_root / "testing" / "label_2").mkdir(parents=True, exist_ok=True)
    return subset_root


def load_coco_annotations(annotations_zip: Path):
    with zipfile.ZipFile(annotations_zip, "r") as archive:
        with archive.open("annotations/instances_val2017.json") as handle:
            return json.load(handle)


def build_coco_subset(output_root: Path, limit: int, timeout: int) -> Path:
    subset_root = output_root / "coco"
    images_root = subset_root / "val2017"
    annotations_root = subset_root / "annotations"
    archive_root = output_root / "_archives"

    annotations_zip = download_file(
        COCO_ANNOTATIONS_URL,
        archive_root / "annotations_trainval2017.zip",
        timeout,
        expect_zip=True,
    )
    try:
        coco = load_coco_annotations(annotations_zip)
    except zipfile.BadZipFile:
        annotations_zip.unlink(missing_ok=True)
        annotations_zip = download_file(
            COCO_ANNOTATIONS_URL,
            archive_root / "annotations_trainval2017.zip",
            timeout,
            expect_zip=True,
        )
        coco = load_coco_annotations(annotations_zip)

    selected_images = sorted(coco["images"], key=lambda item: item["id"])[:limit]
    selected_ids = {image["id"] for image in selected_images}
    selected_annotations = [
        ann for ann in coco["annotations"] if ann["image_id"] in selected_ids
    ]
    used_category_ids = {ann["category_id"] for ann in selected_annotations}
    selected_categories = [
        cat for cat in coco["categories"] if cat["id"] in used_category_ids
    ]

    images_root.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)

    for image in selected_images:
        filename = image["file_name"]
        destination = images_root / filename
        if destination.exists():
            continue
        image_url = image.get("coco_url") or f"{COCO_IMAGE_BASE_URL}/{filename}"
        download_file(image_url, destination, timeout)

    filtered_annotations = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": selected_categories,
    }

    annotation_path = annotations_root / "instances_val2017.json"
    annotation_path.write_text(json.dumps(filtered_annotations), encoding="utf-8")
    return subset_root


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("kitti", "all"):
        subset_root = build_kitti_subset(output_root, args.limit, args.timeout)
        print(f"KITTI subset ready at: {subset_root}")
        print("Note: KITTI is distributed here as full ZIP archives, so the script extracts only the first N images after download.")

    if args.dataset in ("coco", "all"):
        subset_root = build_coco_subset(output_root, args.limit, args.timeout)
        print(f"COCO subset ready at: {subset_root}")


if __name__ == "__main__":
    main()

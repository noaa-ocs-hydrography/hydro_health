import concurrent.futures
import posixpath
import xml.etree.ElementTree as ET
from collections import Counter
import s3fs

# S3 Target Path
S3_TARGET_PATH = "s3://ocs-dev-csdl-hydrohealth/ER_3/model_variables/Prediction/pre_processed/BlueTopo"

MAX_WORKERS = 16


def clean_s3_path(path: str) -> str:
    """Strip s3:// prefix and trailing slashes so s3fs correctly identifies the bucket."""
    path = path.strip()
    if path.startswith("s3://"):
        path = path[7:]
    return path.rstrip("/")


def get_feature_detection_score(row_data: dict, increased_scale: bool = False) -> float:
    """Replicates feature detection scoring logic using row attribute values."""
    feat_detect = bool(int(row_data.get("significant_features", 0)))
    feat_least_depth = bool(int(row_data.get("feature_least_depth", 0)))

    least_depth = feat_detect and feat_least_depth

    if feat_detect and least_depth:
        return 110.0 if increased_scale else 100.0
    else:
        return 60.0


def process_single_xml(fs: s3fs.S3FileSystem, xml_s3_path: str) -> list[float]:
    """Reads and parses a single PAMRasterBand attribute table directly from S3."""
    scores = []
    try:
        with fs.open(xml_s3_path, "rb") as f:
            xml_bytes = f.read()

        root = ET.fromstring(xml_bytes)

        # Locate Contributor Band
        for band in root.findall(".//PAMRasterBand"):
            desc = band.find("Description")
            if desc is not None and desc.text == "Contributor":
                rat_node = band.find(".//GDALRasterAttributeTable")
                if rat_node is None:
                    continue

                field_names = [
                    f.find("Name").text
                    for f in rat_node.findall("FieldDefn")
                    if f.find("Name") is not None
                ]

                for row in rat_node.findall(".//Row"):
                    f_vals = [f.text for f in row.findall("F")]
                    row_data = {
                        field_names[i]: f_vals[i]
                        for i in range(min(len(field_names), len(f_vals)))
                    }

                    # Filter out empty/unassigned survey rows
                    start_date = row_data.get("survey_date_start")
                    end_date = row_data.get("survey_date_end")

                    if start_date or end_date:
                        score = get_feature_detection_score(row_data)
                        scores.append(score)

    except Exception:
        # Silently skip missing or unparseable XML files
        pass

    return scores


def calculate_bucket_feature_scores(target_path: str):
    # Initialize s3fs leveraging EC2 IAM Role credentials automatically
    fs = s3fs.S3FileSystem()

    # base_path = clean_s3_path(target_path)
    glob_pattern = f"{'ocs-dev-csdl-hydrohealth/ER_3/model_variables/Prediction/pre_processed/BlueTopo'}/**/*.aux.xml"
    
    print(f"Searching S3 using pattern: {glob_pattern} ...")

    xml_s3_paths = fs.glob(glob_pattern)

    print(f"Found {len(xml_s3_paths)} XML files matching glob pattern.")

    if not xml_s3_paths:
        print("\n[!] No XML files were found under this prefix.")
        return

    print(f"Processing XML files using multithreading ({MAX_WORKERS} workers)...\n")

    score_counts = Counter()
    total_tiles_processed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {
            executor.submit(process_single_xml, fs, path): path
            for path in xml_s3_paths
        }

        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                tile_scores = future.result()
                if tile_scores:
                    score_counts.update(tile_scores)
                    total_tiles_processed += 1
            except Exception as e:
                print(f"Error processing {posixpath.basename(path)}: {e}")

    # Summary Report
    total_surveys_evaluated = sum(score_counts.values())

    print("\n" + "=" * 50)
    print("FEATURE DETECTION SCORE ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total GeoTIFF Folders/XMLs Evaluated: {total_tiles_processed}")
    print(f"Total Contributor Surveys Analyzed:   {total_surveys_evaluated}")
    print("-" * 50)

    if total_surveys_evaluated > 0:
        for score, count in sorted(score_counts.items()):
            percentage = (count / total_surveys_evaluated) * 100
            print(f"Score {score:>5.1f} : {count:>6} surveys ({percentage:>6.2f}%)")
    else:
        print("No valid contributor survey records were found in the XML attribute tables.")


if __name__ == "__main__":
    calculate_bucket_feature_scores(S3_TARGET_PATH)
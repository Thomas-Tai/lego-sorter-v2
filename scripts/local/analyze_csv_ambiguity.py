import pandas as pd
import os

# Define paths
BASE_DIR = r"C:\D\WorkSpace\[Cloud]_Company_Sync\MSC\OwnInfo\MyResearchProject\Lego Sorter\_2 labelling_software\project_root_demo\2_lego_database\rebrickableToSqlite3\LEGO_Datafiles_20240627"
ELEMENTS_PATH = os.path.join(BASE_DIR, "elements.csv")
PARTS_PATH = os.path.join(BASE_DIR, "parts.csv")
COLORS_PATH = os.path.join(
    BASE_DIR, "colors.csv.gz"
)  # Note: listed as .csv.gz in previous `list_dir`


def analyze_ambiguity():
    print("Loading datasets...")
    try:
        elements_df = pd.read_csv(ELEMENTS_PATH)
        parts_df = pd.read_csv(PARTS_PATH)
        # Handle colors which might be gzipped
        if os.path.exists(COLORS_PATH):
            colors_df = pd.read_csv(COLORS_PATH, compression="gzip")
        else:
            # Fallback if unzipped version exists
            colors_df = pd.read_csv(os.path.join(BASE_DIR, "colors.csv"))
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    print(f"Elements: {len(elements_df)}")
    print(f"Parts: {len(parts_df)}")

    # Group by Part Number and Color to find variations (same part/color, different Element ID)
    print("\nAnalyzing for Ambiguity (Multiple Element IDs for same Part+Color)...")

    # Group by part_num and color_id, count unique element_ids
    grouped = (
        elements_df.groupby(["part_num", "color_id"])["element_id"]
        .nunique()
        .reset_index()
    )
    grouped.columns = ["part_num", "color_id", "variation_count"]

    # Filter for cases with > 1 variation
    ambiguous_parts = grouped[grouped["variation_count"] > 1].sort_values(
        "variation_count", ascending=False
    )

    summary = []
    summary.append(f"Total Part/Color combinations: {len(grouped)}")
    summary.append(f"Ambiguous combinations (Count > 1): {len(ambiguous_parts)}")
    summary.append(
        f"Percentage of ambiguity: {len(ambiguous_parts)/len(grouped)*100:.2f}%"
    )
    summary.append("\n--- Top 20 Most 'Confusing' Parts (Most Variations) ---")

    print("\n".join(summary))

    temp_dir = os.environ.get("TEMP", ".")
    report_path = os.path.join(temp_dir, "analysis_report.txt")

    # Write header
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")

    for index, row in ambiguous_parts.head(20).iterrows():
        part_num = row["part_num"]
        color_id = row["color_id"]
        count = row["variation_count"]

        # Get names
        part_name = (
            parts_df[parts_df["part_num"] == part_num]["name"].values[0]
            if not parts_df[parts_df["part_num"] == part_num].empty
            else "Unknown"
        )
        color_name = (
            colors_df[colors_df["id"] == color_id]["name"].values[0]
            if not colors_df[colors_df["id"] == color_id].empty
            else f"Color {color_id}"
        )

        # Get the list of Element IDs
        ids = elements_df[
            (elements_df["part_num"] == part_num)
            & (elements_df["color_id"] == color_id)
        ]["element_id"].tolist()

        report_lines = []
        report_lines.append(
            f"Part: {part_num} ({part_name}) | Color: {color_name} | Variations: {count}"
        )
        report_lines.append(f"  -> Element IDs: {ids}")
        print("\n".join(report_lines))

        temp_dir = os.environ.get("TEMP", ".")
        report_path = os.path.join(temp_dir, "analysis_report.txt")

        with open(report_path, "a", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")

        print(f"Report written to: {report_path}")


if __name__ == "__main__":
    temp_dir = os.environ.get("TEMP", ".")
    report_path = os.path.join(temp_dir, "analysis_report.txt")
    # Clear previous report
    if os.path.exists(report_path):
        os.remove(report_path)
    analyze_ambiguity()

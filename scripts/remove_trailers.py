import sys
import re


def main():
    # Regex to match the undesired lines (case-insensitive)
    # Matches:
    # Co-authored-by: Name <email>
    # Generated-by: Name
    # Generated with: Tool
    pattern = re.compile(r"^\s*(Co-authored-by|Generated-by|Generated with)\s*:.*", re.IGNORECASE)

    try:
        # Read from stdin
        lines = sys.stdin.readlines()

        filtered_lines = []
        for line in lines:
            # Check if line matches the attribution pattern
            if not pattern.match(line):
                filtered_lines.append(line)

        # Write to stdout
        sys.stdout.writelines(filtered_lines)
    except Exception as e:
        sys.stderr.write(f"Error in filter script: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

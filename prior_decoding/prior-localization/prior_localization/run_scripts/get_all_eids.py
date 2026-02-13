from pathlib import Path


in_file = Path("/Users/changyin/Downloads/region_session_ids.txt")
out_file = Path("/Users/changyin/Downloads/session_ids_unique.txt")

unique_eids = set()


with in_file.open("r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # skip empty lines
        parts = line.split()
        if len(parts) < 2:
            continue  # skip malformed lines
        eid = parts[1]  # second column
        unique_eids.add(eid)

# Write unique eids
with out_file.open("w") as f:
    for eid in sorted(unique_eids):
        f.write(eid + "\n")

print(f"Saved {len(unique_eids)} unique session IDs to:")
print(out_file)

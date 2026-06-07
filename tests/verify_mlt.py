import xml.etree.ElementTree as ET

def verify_mlt_structure(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    seen_ids = set()
    errors = []

    # Check element ordering
    current_stage = 0 # 0: profile, 1: producers/chains, 2: playlists, 3: track tractors, 4: main_bin, 5: project tractor

    stages = {
        "profile": 0,
        "chain": 1,
        "producer": 1,
        "playlist": 2,
        "tractor": 3
    }

    for child in root:
        tag = child.tag
        cid = child.get("id", "")

        # Determine stage
        if tag == "playlist" and cid == "main_bin":
            stage = 4
        elif tag == "tractor" and child.find("property[@name='kdenlive:projectTractor']") is not None:
            stage = 5
        else:
            stage = stages.get(tag, current_stage)

        if stage < current_stage:
            errors.append(f"Element {tag} (id={cid}) is out of order. Stage {stage} appeared after stage {current_stage}")
        current_stage = max(current_stage, stage)

        if cid:
            seen_ids.add(cid)

    # Check for broken references
    for elem in root.iter():
        prod = elem.get("producer")
        if prod and prod not in seen_ids and prod != "black_track":
            errors.append(f"Broken reference: Element {elem.tag} refers to missing producer '{prod}'")

    return errors

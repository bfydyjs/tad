def remove_duplicate_annotations(ants, tol=1e-3):
    # remove duplicate annotations (same category and starting/ending time)
    valid_events = []
    for event in ants:
        s, e = event["segment"]
        label = event["label"]

        # here, we add removing the events whose duration is 0, (HACS)
        if e - s <= 0:
            continue

        is_duplicate = any(
            p["label"] == label
            and abs(s - p["segment"][0]) <= tol
            and abs(e - p["segment"][1]) <= tol
            for p in valid_events
        )

        if not is_duplicate:
            valid_events.append(event)

    return valid_events

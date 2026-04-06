def flatten_map(mapping):
    pairs = [
        (key, kw)
        for key, kws in mapping.items()
        for kw in kws
    ]

    # 긴 키워드 우선
    pairs = sorted(pairs, key=lambda x: len(x[1]), reverse=True)

    return pairs
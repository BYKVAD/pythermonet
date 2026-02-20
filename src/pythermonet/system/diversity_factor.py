def diversity_factor_from_n_heat_pumps(n: int) -> float:
    """
    Samtidighedsfaktor f(n). Skal returnere (0, 1].

    Implementér her din valgte empiriske/standard kurve.
    Nedenstående er en placeholder, som du skal erstatte med din rigtige relation.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    # Placeholder: asymptotisk fald mod et minimum.
    # Udskift med din faktiske model/kurve.
    f_min = 0.62
    k = 0.38

    f = f_min + k/n

    return float(f)